
# Cell [1]
import monai.deploy.core as md  # 'md' stands for MONAI Deploy (or can use 'core' instead)
from monai.deploy.core import (
    Application,
    DataPath,
    ExecutionContext,
    Image,
    InputContext,
    IOType,
    Operator,
    OutputContext,
)
from monai.transforms import (
    AddChannel,
    Compose,
    Lambda,
    LoadImage,
    Resize,
    ScaleIntensity,
)

PNEUMONIA_CLASSES = ["NORMAL", "PNEUMONIA"]

# Cell [2]
@md.input("image", DataPath, IOType.DISK)
@md.output("output", DataPath, IOType.DISK)
@md.env(pip_packages=["monai"])
class PneumoniaClassifierOperator(Operator):
    """Classifies the given image and returns the class name."""

    @property
    def transform(self):
        val_transforms = Compose(
            [
                LoadImage(image_only=True),
                Lambda(func=lambda x: np.mean(x, axis=2) if len(x.shape) >= 3 else x),
                AddChannel(),
                ScaleIntensity(),
                Resize(spatial_size=(224,224)),
            ]
        )
        return val_transforms

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        import json

        import torch
        
        input_path = op_input.get().path
        if input_path.is_dir():
            input_path = next(input_path.glob("*.*"))  # take the first file

        image_tensor = self.transform(input_path)  # (1, 224, 224), torch.float64
        image_tensor = image_tensor[None].float()  # (1, 1, 224, 224), torch.float32

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)

        model = context.models.get()  # get a TorchScriptModel object

        with torch.no_grad():
            outputs = model(image_tensor)

        _, output_classes = outputs.max(dim=1)

        result = PNEUMONIA_CLASSES[output_classes[0]]  # get the class name
        print(result)

        # Get output (folder) path and create the folder if not exists
        output_folder = op_output.get().path
        output_folder.mkdir(parents=True, exist_ok=True)

        # Write result to "output.json"
        output_path = output_folder / "output.json"
        with open(output_path, "w") as fp:
            json.dump(result, fp)
            
# Cell [3]
@md.resource(cpu=1, gpu=1, memory="1Gi")
class App(Application):
    """Application class for the Pneumonia classifier."""

    def compose(self):
        classifier_op = PneumoniaClassifierOperator()

        self.add_operator(classifier_op)

# Finally
if __name__ == "__main__":
    App(do_run=True)
