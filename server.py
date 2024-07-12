from ultralytics import YOLO
import cv2
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import threading
import time
import asyncio
import websockets


# download and load the models being used
def setup():
    # download and load the object detection model
    od_model = YOLO("yolov8n")
    # download and load the large language model
    model_id = "llava-hf/llava-1.5-7b-hf"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    ll_processor = AutoProcessor.from_pretrained(model_id)
    ll_model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    return od_model, ll_model, ll_processor


# run the object detection model on the video
def objectdetection_model(video_stream, od_model):
    od_model.predict(
        source=video_stream,
        show=True,              # open window with live labels
        vid_stride=2,           # skip every other frame for speed
        verbose=False,          # don't print outputs
        device='cpu',           # large language model gets to use gpu
    )
    return


# run the large language model on video frames
def largelanguage_model(video_stream, ll_model, ll_processor):
    # need metadata to get the active frame of the video
    start_time = time.time()
    cap = cv2.VideoCapture(video_stream)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # define inner function to send the live textual updates
    async def send_text(websocket, path):
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, (time.time()-start_time)*fps-1)
            res, frame = cap.read()
            # prompt and receive output from the large language model
            prompt = "USER: You are a safety inspector tasked with analyzing a construction site for potential safety hazards. Review each image provided, identify and describe all hazards, explain why each is a hazard, and suggest ways to eliminate or reduce the danger of each. The goal is to make the construction site safer and prevent incidents.\nRespond clearly, concisely, and professionally only in the following format:\n**Hazard**: [Description of the hazard]\n**Explanation**: [Brief explanation of why it is a hazard]\n**Suggestion**: [Suggestion to eliminate or reduce the danger]\nDo not repeat previously reported hazards. If no hazards exist, respond only with \"**NONE**\".\n\nUSER: <image>\n\nASSISTANT: "
            inputs = ll_processor(prompt, images=[frame], padding=True, return_tensors="pt").to("cuda")
            output = ll_model.generate(**inputs, max_new_tokens=512)
            generated_text = ll_processor.batch_decode(output, skip_special_tokens=True)
            message = generated_text[0].split("ASSISTANT: \n")[1].strip()
            await websocket.send(message)

    print("Please reload your browser")
    # start running server so that messages can be sent
    start_server = websockets.serve(send_text, "localhost", 6789)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
    return


# run the entire assistant
def main():
    video_stream = "../../360.mp4"
    # run preliminaries so application can be run
    od_model, ll_model, ll_processor = setup()
    # set up video as a separate thread
    objectdetection_t = threading.Thread(target=objectdetection_model, args=(video_stream, od_model,))
    objectdetection_t.start()
    # run language modelling on the main thread for website communication
    largelanguage_model(video_stream, ll_model, ll_processor)
    return


if __name__ == "__main__":
    main()
