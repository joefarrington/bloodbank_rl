from tianshou.policy import A2CPolicy
from typing import Any, Dict, List, Optional, Type
from tianshou.data import Batch

# Normal key structure for output doesn't work with MLFlow logging nicely
# loss used as a key and also parent


class A2CPolicyforMLFlow(A2CPolicy):
    def learn(
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        loss_dict = super().learn(batch, batch_size, repeat, **kwargs)
        output_loss_dict = {}
        output_loss_dict["loss"] = loss_dict["loss"]
        output_loss_dict["loss_component/actor"] = loss_dict["loss/actor"]
        output_loss_dict["loss_component/vf"] = loss_dict["loss/vf"]
        output_loss_dict["loss_component/ent"] = loss_dict["loss/ent"]
        return loss_dict
