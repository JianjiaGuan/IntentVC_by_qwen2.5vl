import basic_utils as utils
import os

user_content_template = "<video> Describe this video with the intention-oriented object: {}.The following is a reference format for two descriptive sentences: 1.a [color] [object] [action] in [environmental context]. 2.the [object] [performs action] while [surrounding context]."
# user_content_template = "Describe this video with the intention-oriented object: {}. Here are a few examples for reference, please keep the style consistent:\n\n"

if __name__ == '__main__':
    data = utils.load_json('data/mllm_video.json')
    for item in data:
        video_obj_cls = os.path.basename(item['videos'][0]).split('-')[0]
        user_content = user_content_template.format(video_obj_cls)
        item['messages'][0]['content'] = user_content
    
    utils.save_json(data, "data/mllm_video_with_obj_class.json", save_pretty=True)