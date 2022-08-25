
# Cell [1]
import monai.deploy.core as md  # 'md' stands for MONAI Deploy (or can use 'core' instead)
from monai.deploy.core import (
    Application,
    DataPath,
    ExecutionContext,
    InputContext,
    IOType,
    Operator,
    OutputContext,
)
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    BoundingRectd,
    EnsureChannelFirstd,
    EnsureTyped,
    Compose,
    CropForegroundd,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    ToTensord,
)
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)

import os
import glob
import torch

# Cell [2]
@md.input("image", DataPath, IOType.DISK)
@md.output("output", DataPath, IOType.DISK)
@md.env(pip_packages=["monai"])
class SpleenDetectorOperator(Operator):
    """Classifies the given image and returns the class name."""

    @property
    def transform(self):
        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                EnsureTyped(keys=["image"], dtype=torch.float32),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-57, a_max=164,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear")),
            ]
        )
        return test_transforms

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        import json

        import torch
        
        input_path = op_input.get().path
        if input_path.is_dir():
            input_path = next(input_path.glob("*.*"))  # take the first file

        image_tensor = self.transform({"image": input_path})  # Load path as dict
        image_tensor = torch.tensor(image_tensor["image"].array)  # Get just the image for input

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = [image_tensor.to(device)]

        ### Build Anchor Generator
        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=[2**l for l in range(len([1,2]) + 1)],
            base_anchor_shapes=[[6,8,4],[8,6,5],[10,10,6]],
        )
        
        ### Build Model
        model = context.models.get()  # get a TorchScriptModel object
        model.predictor = torch.jit.load(model.path, map_location=device).eval()
        net = model.predictor
        
        ### Build Detector
        detector = RetinaNetDetector(
            network=net, anchor_generator=anchor_generator, debug=False
        ).to(device)

        detector.set_target_keys(box_key="label_box", label_key="label_class")

        # set validation components
        detector.set_box_selector_parameters(
            score_thresh=0.02,
            topk_candidates_per_level=1000,
            nms_thresh=0.22,
            detections_per_img=100,
        )
        detector.set_sliding_window_inferer(
            roi_size=[512,512,208],
            overlap=0.25,
            sw_batch_size=1,
            mode="constant",
            device="cpu",
        )
        
        detector.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                val_outputs_all = detector(image_tensor, use_inferer=True)

        pred_boxes=[
            val_data_i[detector.target_box_key].cpu().detach().numpy().tolist()
            for val_data_i in val_outputs_all
        ],
        pred_classes=[
            val_data_i[detector.target_label_key].cpu().detach().numpy().tolist()
            for val_data_i in val_outputs_all
        ],
        pred_scores=[
            val_data_i[detector.pred_score_key].cpu().detach().numpy().tolist()
            for val_data_i in val_outputs_all
        ],
        print(pred_boxes, pred_classes, pred_scores)
        result = {"boxes": pred_boxes, "classes": pred_classes, "scores": pred_scores}

        # Get output (folder) path and create the folder if not exists
        output_folder = op_output.get().path
        output_folder.mkdir(parents=True, exist_ok=True)

        # Write result to "output.json"
        output_path = output_folder / "output.json"
        with open(output_path, "w") as fp:
            json.dump(result, fp)
            
# Cell [3]
@md.resource(cpu=1, gpu=1, memory="6Gi")
class App(Application):
    """Application class for the Spleen detector."""

    def compose(self):
        classifier_op = SpleenDetectorOperator()

        self.add_operator(classifier_op)

# Finally
if __name__ == "__main__":
    App(do_run=True)
