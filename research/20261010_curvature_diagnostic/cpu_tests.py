"""Run meaningful shared-core fixtures and official TRAIN loader on MMRec CPU."""
import json
from pathlib import Path
import audit

HERE = Path(__file__).resolve().parent
TESTS = r'''
import hashlib, json, math, os, random, sys, unittest
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='4'
import numpy as np
import scipy.sparse as sp
import torch
torch.set_num_threads(4)
''' + audit.CORE + r'''

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_trs=torch.nn.Linear(2,1,bias=False).double()
        self.gate_v=torch.nn.Linear(1,1,bias=False).double()
        self.text_trs=torch.nn.Linear(2,1,bias=False).double()
        self.gate_t=torch.nn.Linear(1,1,bias=False).double()
        self.query_common=torch.nn.Linear(2,1,bias=False).double()
        self.reg_weight=.25
        self.fail=False
    def cal_bpr_loss(self,value):
        return value.square().mean()
    def cal_reg_loss(self):
        return sum(p.square().sum() for p in self.parameters())
    def calculate_loss(self,batch):
        value=self.gate_v(self.image_trs(batch))+self.gate_t(self.text_trs(batch))
        bpr=self.cal_bpr_loss(value)
        reg=self.cal_reg_loss()
        if self.fail:raise RuntimeError('injected forward error')
        return bpr+.03*self.query_common(batch).square().mean()+self.reg_weight*reg

class Tests(unittest.TestCase):
    def test_exact_component_and_gradient_parity(self):
        model=Tiny();batch=torch.tensor([[.2,.4],[.1,.8]],dtype=torch.float64)
        original=model.calculate_loss(batch)
        g=torch.autograd.grad(original,list(model.parameters()))
        hooked=component_tensors(model,batch)
        h=torch.autograd.grad(hooked['full'],list(model.parameters()))
        self.assertTrue(torch.equal(original,hooked['full']))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(g,h)))
        self.assertNotIn('cal_bpr_loss',model.__dict__)
        self.assertNotIn('cal_reg_loss',model.__dict__)
    def test_hook_cleanup_on_exception(self):
        model=Tiny();model.fail=True
        with self.assertRaises(RuntimeError):component_tensors(model,torch.ones(1,2,dtype=torch.float64))
        self.assertNotIn('cal_bpr_loss',model.__dict__)
        self.assertNotIn('cal_reg_loss',model.__dict__)
    def test_probe_restoration_even_on_failure(self):
        model=Tiny();batch=torch.ones(2,2,dtype=torch.float64);before=clone_state(model)
        params=named_block(model,'image');direction=private_gaussian(params,20261020)
        z=losses(model,batch);model.fail=True
        with self.assertRaises(RuntimeError):probe(model,batch,params,direction,before,z,z,.01)
        self.assertTrue(exact_state(model,before))
    def test_quadratic_symmetric_regularizer(self):
        model=Tiny();batch=torch.zeros(2,2,dtype=torch.float64);before=clone_state(model)
        params=named_block(model,'image');unit=private_gaussian(params,20261020)
        zero=losses(model,batch);values,dsq,valid=probe(model,batch,params,unit,before,zero,zero,.1)
        self.assertAlmostEqual(values['regularizer']['E'],model.reg_weight*dsq,places=11)
        self.assertAlmostEqual(values['regularizer']['K'],2*model.reg_weight,places=11)
        self.assertTrue(valid);self.assertTrue(exact_state(model,before))
    def test_private_gaussian_rng_and_alignment(self):
        model=Tiny();params=named_block(model,'image');before=torch.get_rng_state().clone()
        a=private_gaussian(params,20261020);b=private_gaussian(params,20261020)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))
        self.assertAlmostEqual(l2(a),1.,places=12)
        self.assertAlmostEqual(cosine(a,a),1.,places=12)
    def test_private_train_universe_sampler(self):
        users=np.array([0,0,1,1,2,2],dtype=np.int64);pos=np.array([0,1,1,2,0,2],dtype=np.int64)
        csr=sp.csr_matrix((np.ones(6),(users,pos)),shape=(3,4));csr.sort_indices()
        before=np.random.get_state();a,chosen,u=fixed_private_batches(users,pos,csr,batch_size=2,n_batches=3)
        b,chosen2,u2=fixed_private_batches(users,pos,csr,batch_size=2,n_batches=3)
        self.assertTrue(np.array_equal(a,b));self.assertEqual(len(np.unique(chosen)),6)
        self.assertTrue(np.array_equal(before[1],np.random.get_state()[1]))
        self.assertTrue(set(a[:,2,:].ravel()).issubset(set(u)))
        self.assertEqual(a.shape,(3,3,2))
    def test_fixed_gate_rejects_state_disagreement_and_invalid(self):
        def rows(reverse=False):
            r=[]
            for s in ('init','uniform_best'):
                for b in range(3):
                    for d in ('full_grad','rand0','rand1','rand2'):
                        for block in ('image','text'):
                            winner='text' if reverse and s=='uniform_best' else 'image'
                            v=.1 if block==winner else .01
                            r.append({'state':s,'batch_id':b,'direction':d,'block':block,
                                      'numerically_valid':True,
                                      'components':{q:{'E':v,'K':v,'floor_E':1e-5,'floor_K':1e-5}
                                                    for q in ('bpr','full')}})
            return r
        self.assertTrue(dominance(rows())['passed'])
        self.assertFalse(dominance(rows(True))['passed'])
        invalid=rows();invalid[0]['numerically_valid']=False
        self.assertFalse(dominance(invalid)['passed'])
    def test_numerical_floor_signed_excess(self):
        m=measurement({'bpr':1.},{'bpr':1.},{'bpr':1.2},{'bpr':.7},{'bpr':1.},.01)['bpr']
        self.assertLess(m['E'],0);self.assertAlmostEqual(m['K'],-10)
        self.assertAlmostEqual(m['floor_E'],1.2e-5)
    def test_official_cpu_loader_and_checkpoint_schema_without_forward(self):
        root='/data/sunzixuan/interest_negative_20261010_v1'
        os.chdir(root+'/carrier/src');sys.path[:0]=[root,root+'/carrier/src']
        from utils.configurator import Config
        from utils.dataset import RecDataset
        from utils.dataloader import TrainDataLoader
        from utils.utils import init_seed
        from models.msca import MSCA
        settings=json.load(open(root+'/runs/baby_uniform_s2026_formal/config.json'))
        settings['use_gpu']=False
        config=Config('MSCA','baby',settings,mg=False)
        data=RecDataset(config)
        train=data.df.loc[data.df['x_label']==0].drop(columns=['x_label']).copy()
        train_data=data.copy(train);str(train_data)
        loader=TrainDataLoader(config,train_data,batch_size=2048,shuffle=True)
        self.assertEqual(config['device'].type,'cpu')
        self.assertEqual(len(train),118551)
        arrays,chosen,universe=fixed_private_batches(train[data.uid_field].to_numpy(dtype=np.int64),
                       train[data.iid_field].to_numpy(dtype=np.int64),loader.inter_matrix(form='csr'))
        self.assertEqual(arrays.shape,(3,3,2048));self.assertEqual(len(np.unique(chosen)),6144)
        checkpoint=torch.load(root+'/runs/baby_uniform_s2026_formal/best.pt',map_location='cpu')
        self.assertIn('state_dict',checkpoint);self.assertIn('initial_sha256',checkpoint)
        self.assertEqual(checkpoint['epoch'],44)
        self.assertFalse(torch.cuda.is_initialized())

suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
result=unittest.TextTestRunner(verbosity=2).run(suite)
receipt={'status':'pass' if result.wasSuccessful() else 'fail','tests_run':result.testsRun,
         'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
         'cuda_initialized':torch.cuda.is_initialized(),'torch':torch.__version__,'numpy':np.__version__}
print('CPU_TEST_RESULT='+json.dumps(receipt),flush=True)
if not result.wasSuccessful():raise SystemExit(1)
'''

def main():
    compile(TESTS,'<cpu-fixtures>','exec')
    code,output=audit.invoke(TESTS,600)
    marker='CPU_TEST_RESULT='
    start=output.find(marker)
    if start<0:raise RuntimeError(output)
    value,_=json.JSONDecoder().raw_decode(output[start+len(marker):])
    value['source_sha256']=audit.hashlib.sha256((HERE/'audit.py').read_bytes()).hexdigest()
    (HERE/'cpu_tests_receipt.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(output)
    if code:raise SystemExit(code)

if __name__=='__main__':
    main()
