# parnet_additional_utils

Utility library for loading and running ParNet models.

## Contents

### `ParnetModelName`

`str`-based enum of all supported model identifiers.
Accepts string construction for backward compatibility: `ParnetModelName("parnet_new.11m-5.0")`.

```python
from parnet_additional_utils import ParnetModelName

name = ParnetModelName.PARNET_NEW_11M_5_0   # "parnet_new.11m-5.0"
name = ParnetModelName("parnet.7m-2.5")     # also valid
```

### `load_parnet_model`

Load a ParNet checkpoint and apply the correct post-load configuration for the given model variant
(projection reset, `use_maximum_target_control_logprob`, `control_nograd`).

```python
from parnet_additional_utils import ParnetModelName, load_parnet_model
import torch

model = load_parnet_model(
    parnet_model_name=ParnetModelName.PARNET_NEW_11M_5_0,
    filepath="/path/to/weights.pt",
    dtype=torch.float32,
    device=torch.device("cpu"),
)
```

Also accepts a plain string for the model name.

### `ParnetInputDict`

`dict` subclass that enforces a single `"sequences"` key — guarantees ParNet's expected input
contract at construction time.

```python
from parnet_additional_utils import ParnetInputDict

x = ParnetInputDict(sequences_tensor)
x["sequences"]   # ok
x["other"]       # raises KeyError
```

### `parnet_predict`

Thin wrapper around model inference that routes the input format based on model version
(`model_metadata["version"]`): version `"0.5.0"` passes raw sequences; all other versions
pass the full `ParnetInputDict`.

## Supported models

| Enum member | Model string |
| --- | --- |
| `PARNET_NEW_11M_5_0` | `parnet_new.11m-5.0` |
| `PARNET_21M_NONE` | `parnet.21m-none` |
| `PARNET_21M_0_0` | `parnet.21m-0.0` |
| `PARNET_21M_5_0` | `parnet.21m-5.0` |
| `PARNET_7M_0_0` | `parnet.7m-0.0` |
| `PARNET_7M_2_5` | `parnet.7m-2.5` |
| `PARNET_7M_10_0` | `parnet.7m-10.0` |
| `PARNET_7M_20_0` | `parnet.7m-20.0` |
| `PARNET_7M_80_0` | `parnet.7m-80.0` |
