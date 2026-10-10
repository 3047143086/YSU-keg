Read PROTOCOL.md. Work only in this isolated audit directory. Protect the v1
baseline source, queue, inputs, failure packet and checkpoint. First derive and
test the equivalent mapping and its VJP on small CPU examples. Commit the
protocol and source and request root review. Only after that review run the
fixed bounded replay audit. Never run training or a GPU task under this audit.
Record numerical or implementation failures as crash, not innovation outcomes.
