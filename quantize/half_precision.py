import time
import torch
import torchvision

def calc_time(model, image, wrong_method=False):
    # warming up
    with torch.no_grad():
        output = model(image)

    # calculate elapsed time
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        output = model(image)
    if not wrong_method:
        torch.cuda.synchronize()
    end = time.time()
    elapsed_time = end - start

    return elapsed_time

def main():
    # load FP32 resnet50 model
    model = torchvision.models.resnet50()
    model = model.to("cuda")
    model.eval()

    # make dummy image
    image = torch.ones((512, 3, 224, 224))
    image = image.to("cuda")

    fp32_elapsed_time = calc_time(model, image)
    fp32_elapsed_time_wrong = calc_time(model, image, wrong_method=True)

    # load FP16 resnet50 model
    model = model.half()
    image = image.half()

    fp16_elapsed_time = calc_time(model, image)
    fp16_elapsed_time_wrong = calc_time(model, image, wrong_method=True)

    print(f"FP32 running time: {fp32_elapsed_time}")
    print(f"FP16 running time: {fp16_elapsed_time}")
    print(f"Wrong FP32 running time: {fp32_elapsed_time_wrong}")
    print(f"Wrong FP16 running time: {fp16_elapsed_time_wrong}")

if __name__=="__main__":
    main()
