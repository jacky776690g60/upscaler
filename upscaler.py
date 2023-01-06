"""
Image Upscaler
Require Python3+

OpenCV super resolution only works on 4 
specific models, and these upscaling 
models can be found here.

Links:
https://github.com/Saafke/EDSR_Tensorflow
https://github.com/fannymonori/TF-ESPCN
https://github.com/Saafke/FSRCNN_Tensorflow
https://github.com/fannymonori/TF-LAPSRN
"""

import sys, os, argparse, time
import cv2
from progressbar import ProgressBar, TermArtist


def upscale():
    """
    Upscale Images
    """
    parser = argparse.ArgumentParser(description="Image Upscaling Tool")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("-i", "--image", nargs="+", type=str, help="Path to the low-res image(s)")
    grp.add_argument("-d", "--img_dir", type=str, help="Path to the low-res image(s)")
    args = parser.parse_args()

    mdl_idx = int(input("Which upscaling model to use:\n1. EDSR_x4\n2. ESPCN_x4\n3. FSRCNN_x3\n4. LapSRN_x8\n"))
    if not 1 <= mdl_idx <= 4:
        raise ValueError("Incorrect Input")

    MDL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models",
                            "EDSR_x4.pb" if mdl_idx == 1 else
                            "ESPCN_x4.pb" if mdl_idx == 2 else
                            "FSRCNN_x3.pb" if mdl_idx == 3 else
                            "LapSRN_x8.pb")

    modelName   = MDL_PATH.split(os.path.sep)[-1].split("_")[0].lower()
    scaleFactor = int(MDL_PATH[-4:].replace('x', "").replace(".pb", ""))


    # =============================
    # Init OpenCV's DNN obj
    # =============================
    print(f"[INFO] loading model: {MDL_PATH}")
    print(f"[INFO] model name: {modelName}\tmodel scale: {scaleFactor}")
    sr = cv2.dnn_superres.DnnSuperResImpl_create() # Super Resolution Object
    sr.readModel(MDL_PATH) 
    sr.setModel(modelName, scaleFactor) 
    # print(help(sr.setModel))


    # =============================
    # Load Image(s)
    # =============================
    img_queue = []
    if args.image:
        img_queue = args.image
    else:
        if not os.path.exists(args.img_dir):
            raise FileNotFoundError("Provided path does not exist")

        ext_list = ["bmp", "jpeg", "jpg", "jpe", "jp2", "tiff", "tif", "png"]

        for path, dirs, files in os.walk(args.img_dir):
            for f in files:
                if any(f.endswith(s) for s in ext_list):
                    img_queue.append(os.path.join(path, f))
            

    progress_bar = ProgressBar(len(img_queue), bar_len=50, style=3)
    for i, img in enumerate(img_queue):
        try:
            imgName, extension  = tuple(elem for elem in os.path.basename(img).split('.'))
            print(f"\r[INFO] Processing {imgName}.{extension}", flush=True)
            in_img  = cv2.imread(img)
            t1      = time.time()
            progress_bar.draw(i)
            out_img = sr.upsample(in_img)
            t2      = time.time()
            print()
            print(f"[INFO] (w, h): ({in_img.shape[1]}, {in_img.shape[0]}) -> ({out_img.shape[1]}, {out_img.shape[0]})")
            print(f"[INFO] Time Elapsed: {t2 - t1:.6f} seconds{TermArtist.WHITE}")
            print()

            # =============================
            # Save to disk
            # =============================
            SAVE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
            if not os.path.exists(SAVE_PATH):
                os.mkdir(SAVE_PATH)
            SAVE_PATH = os.path.join(SAVE_PATH, imgName + "_upscaled" + "." + extension)

            cv2.imwrite(SAVE_PATH, out_img)
            
        except Exception as e:
            print(f"{TermArtist.WHITE}[ERROR] Something went wrong processing {imgName}\n{e}")
            sys.exit()
        except KeyboardInterrupt:
            print(TermArtist.WHITE)
    del progress_bar


if __name__ == "__main__":
    upscale()