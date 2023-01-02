"""
Image Upscaler
Required Python3+

OpenCV super resolution only works on 4 
specific models, and these Upscaling 
models can be found here.

Links:
https://github.com/Saafke/EDSR_Tensorflow
https://github.com/fannymonori/TF-ESPCN
https://github.com/Saafke/FSRCNN_Tensorflow
https://github.com/fannymonori/TF-LAPSRN
"""

import sys, os, argparse, time
import cv2


parser = argparse.ArgumentParser(description="Image Upscaling Tool")
parser.add_argument("-m", "--model", type=str, required=True, help="Path to the upscaling model")
parser.add_argument("-i", "--image", nargs="+", type=str, required=True, help="Path to the low-res image(s)")
args = parser.parse_args()


modelName   = args.model.split(os.path.sep)[-1].split("_")[0].lower()
scaleFactor = int(args.model[-4:].replace('x', "").replace(".pb", ""))


# =============================
# Init OpenCV's DNN obj
# =============================
print(f"[INFO] loading model: {args.model}")
print(f"[INFO] model name: {modelName}\tmodel scale: {scaleFactor}")
sr = cv2.dnn_superres.DnnSuperResImpl_create() # Super Resolution Object
sr.readModel(args.model) 
sr.setModel(modelName, scaleFactor) 
# print(help(sr.setModel))


# =============================
# Load Image
# =============================
for img in args.image:
    try:
        imgName, extension  = tuple(elem for elem in os.path.basename(img).split('.'))

        print(f"\033[96m[INFO] Processing {imgName}.{extension}")
        in_img  = cv2.imread(img)
        t1      = time.time()
        out_img = sr.upsample(in_img)
        t2      = time.time()
        print(f"[INFO] (w, h): ({in_img.shape[1]}, {in_img.shape[0]}) -> ({out_img.shape[1]}, {out_img.shape[0]})")
        print(f"[INFO] Time Elapsed: {t2 - t1:.6f} seconds\033[0m")

        # =============================
        # Save to disk
        # =============================
        SAVE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)
        SAVE_PATH = os.path.join(SAVE_PATH, imgName + "_upscaled" + "." + extension)

        cv2.imwrite(SAVE_PATH, out_img)
        print(f"\033[92m[INFO] Saved to:\n{SAVE_PATH}\033[0m")
    except Exception as e:
        print(f"\033[0m[ERROR] Something went wrong processing {imgName}\n{e}")
        sys.exit()