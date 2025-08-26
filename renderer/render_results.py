from renderer.myrenderer import MyRenderer
import os
import hydra
import json
import bpy
import numpy as np
import blendertoolbox as bt


def render_results(cfg, renderer: MyRenderer):
    save_dir = cfg.renderer.output_path
    
    sampled_files = renderer.sample_data_files() # list of IDs correspond to folders

    camera_positions = renderer.generate_camera_positions(num_frames=cfg.renderer.rotation_endframes, radius=np.sqrt(cfg.renderer.camera_kwargs.camPos[0]**2 + cfg.renderer.camera_kwargs.camPos[1]**2), height=cfg.renderer.camera_kwargs.camPos[2])
    saved = 0
    for file in sampled_files:
        transformation, gt_transformation, acc, init_pose = renderer.load_transformation_data(file)

        # if float(acc) > 0.5 or transformation.shape[1] < 4 :
        #     continue

        # if saved > 10:
        #     break

        if not cfg.renderer.jigsaw:
            last_entry = np.expand_dims(transformation[-1], axis=0)
            last_entry = np.repeat(last_entry, cfg.renderer.rotation_endframes, axis=0)
            transformation = np.concatenate([transformation, last_entry], axis=0)
        
        parts = renderer.load_mesh_parts(file)
        
        save_path = f"./BlenderToolBox_render/{save_dir}/{file}_acc{round(float(acc), 3)}"
        os.makedirs(save_path, exist_ok=True)

        imgs_path = os.path.join(save_path, "imgs")
        os.makedirs(imgs_path, exist_ok=True)

        renderer.save_img(parts, gt_transformation, gt_transformation, init_pose, os.path.join(save_path, "gt.png"))

        if not cfg.renderer.jigsaw:
            if not cfg.renderer.only_end:
                frame = 0

                for i in range(transformation.shape[0]):

                    if i >= transformation.shape[0] - cfg.renderer.rotation_endframes:
                        renderer.cam = bt.setCamera(
                            camLocation=camera_positions[i - (transformation.shape[0] - cfg.renderer.rotation_endframes)],
                            lookAtLocation=cfg.renderer.camera_kwargs.camLookat,
                            focalLength=cfg.renderer.camera_kwargs.focalLength
                        )

                    renderer.save_img(parts, gt_transformation, transformation[i], init_pose, os.path.join(imgs_path, f"{frame:04}.png"))

                    frame += 1

                renderer.save_video(imgs_path=imgs_path, video_path=os.path.join(save_path, "video.mp4"), frame=frame)
                renderer.clean()
            else:
                renderer.save_img(parts, gt_transformation, transformation[-1], init_pose, os.path.join(imgs_path, "pred_end.png"))
                renderer.save_img(parts, gt_transformation, transformation[1], init_pose, os.path.join(imgs_path, "init.png"))
                renderer.clean()
        else:
            renderer.save_img(parts, gt_transformation, transformation, init_pose, os.path.join(imgs_path, "pred_end.png"))
            renderer.clean()
        
        saved += 1

        

@hydra.main(config_path="../config", config_name="eval")
def main(cfg):
    renderer = MyRenderer(cfg)
    render_results(cfg, renderer)
    #renderer.save_video(imgs_path="/home/cederic/dev/puzzlefusion-plusplus/BlenderToolBox_render/color_bonesAdvanced_PF++_zeroshot/952_acc0.062/imgs", video_path=os.path.join("/home/cederic/dev/puzzlefusion-plusplus/BlenderToolBox_render/color_bonesAdvanced_PF++_zeroshot/952_acc0.062", "video.mp4"), frame=35)



if __name__ == "__main__":
    main()