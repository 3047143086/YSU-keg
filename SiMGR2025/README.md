from src.models.simgr import SiMGR

## SiMGR Sign-aware Multi-modal Recommendation

#### Requirments

pytorch 2.2.0 

pytorch-cuda  12.1

torch-geometric 2.6.1

Go to the ./src directory, conduct following scripts
#### For baby dataset
```python
python main.py -m SiMGR -d baby
```

#### For sports dataset
```python
python main.py -m SiMGR -d sports
```
#### For electronics dataset
```python
python main.py -m SiMGR -d elec
```