import ai_edge_torch
import edgeimpulse as ei
# import onnx
import torch
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights

ei.API_KEY = 'ei_aee33e8afd18536ac8f9ec8c528a323320d80d2024c3f71ca6b3c0b7508d3393'

print(ei.model.list_profile_devices())

model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
sample_inputs = (torch.randn(1, 3, 224, 224),)
model_path = 'deployment/mobilenetv2'

edge_model = ai_edge_torch.convert(model.eval(), sample_inputs)
edge_model.export(f"{model_path}.tflite")


# onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)
# onnx_program.optimize()
# onnx_path = 'deployment/mobilenetv2.onnx'
# onnx_program.save(onnx_path)

profile_nicla_vision = ei.model.profile(model=model_path, device='arduino-nicla-vision')
print(profile_nicla_vision.summary())

profile_nicla_vision_m4 = ei.model.profile(model=model_path, device='arduino-nicla-vision-m4')
print(profile_nicla_vision_m4.summary())

profile_esp32 = ei.model.profile(model=model_path, device='espressif-esp32')
print(profile_esp32.summary())

profile_stm32 = ei.model.profile(model=model_path, device='st-stm32n6')
print(profile_stm32.summary())

profile_raspbery_pi_4 = ei.model.profile(model=model_path, device='raspberry-pi-4')
print(profile_raspbery_pi_4.summary())