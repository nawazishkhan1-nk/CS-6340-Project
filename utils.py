import json
import glob
import os
import random
import logging
import os
import queue
import shutil
# import torch

def train_eval_split(path_dir):
    all_files = glob.glob(f'{path_dir}/*.story')
    unique_story_ids = set([fn.split('/')[-1].split('.')[0] for fn in all_files])
    random.seed(42)
    train_test_split = 0.85
    train_nums = int(train_test_split * len(unique_story_ids))
    print(f'{train_nums} Training Stories and {len(unique_story_ids) - train_nums} Eval Stories')
    
    train_stories = set(random.sample(unique_story_ids, train_nums))
    test_stories = unique_story_ids - train_stories
    par_dir = path_dir.split('/')
    par_dir = "/".join(par_dir[0:len(par_dir)-1])
    for s_id in train_stories:
        file_types = ['story', 'questions', 'answers']
        for ftype in file_types:
            src_path = f'{path_dir}/{s_id}.{ftype}'
            dest_path = f'{par_dir}/data/train/{s_id}.{ftype}'
            shutil.copyfile(src_path, dest_path)
    for s_id in test_stories:
        file_types = ['story', 'questions', 'answers']
        for ftype in file_types:
            src_path = f'{path_dir}/{s_id}.{ftype}'
            dest_path = f'{par_dir}/data/eval/{s_id}.{ftype}'
            shutil.copy2(src_path, dest_path)
                   
def get_story(story_path):
    with open(story_path) as f:
        all_lines = [line.strip() for line in f.readlines()]
        all_lines = list(filter(lambda x:len(x) > 0, all_lines))
    assert all_lines[0].startswith('HEADLINE: ')
    title = all_lines[0].split('HEADLINE: ')[-1]
    assert all_lines[3] == "TEXT:"
    context = " ".join(all_lines[4:])
    return title, context

def get_question_answer_pairs(path_dir, story_id, context):
    q_file = f'{path_dir}/{story_id}.questions'
    a_file = f'{path_dir}/{story_id}.answers'
    with open(q_file, 'r') as f:
        all_question_lines = [line.strip() for line in f.readlines()]
        
    difficulty_ar = list(map(lambda x: x.split(': ')[-1], list(filter(lambda x:len(x) > 0 and x.startswith('Difficulty:'), all_question_lines))))
    question_ids =  list(map(lambda x: x.split(': ')[-1], list(filter(lambda x:len(x) > 0 and x.startswith('QuestionID:'), all_question_lines))))
    questions =  list(map(lambda x: x.split(': ')[-1], list(filter(lambda x:len(x) > 0 and x.startswith('Question:'), all_question_lines))))
    if os.path.exists(a_file):
        with open(a_file, 'r') as f:
            all_answer_lines = [line.strip() for line in f.readlines()]
            answers =  list(map(lambda x: x.split(': ')[-1], list(filter(lambda x:len(x) > 0 and x.startswith('Answer:'), all_answer_lines))))
    else:
        answers = [[]] * len(questions)
    assert len(answers) == len(question_ids)
    assert len(difficulty_ar) == len(question_ids) == len(questions)
    pairs_ar = []
    for i in range(len(questions)):
        cur_dict = {}
        cur_dict['question'] = questions[i]
        cur_dict['id'] = question_ids[i]
        # cur_dict['is_impossible'] = False if difficulty_ar[i].lower() in ['easy', 'moderate'] else True
        cur_dict['is_impossible'] = False
        ans_ar = []
        if len(answers[i]) > 0:
            all_answers = answers[i].split(' | ')
            for ans in all_answers:
                ans_dict = {}
                ans_dict['text'] = ans
                exact_find = context.lower().find(ans.strip().lower())
                if exact_find == -1:
                    # find in chunks of 4, 3, 2 and so on:
                    res = -1
                    for cs in range(10, 0, -1):
                        ans_temp_ar = ans.split()
                        sz = len(ans_temp_ar)
                        st = 0
                        while (st+cs) < sz:
                            check = " ".join(ans_temp_ar[st:(st+cs)])
                            if context.lower().find(check.lower()) != -1:
                                res = context.lower().find(check.lower())
                                break
                            st += 1
                        if res != -1:
                            break
                    exact_find = res
                ans_dict['answer_start'] = exact_find
                if exact_find == -1 and len(all_answers) > 1:
                    continue
                else:
                    ans_ar.append(ans_dict)
        cur_dict['answers'] = ans_ar
        if ans_ar[0]['answer_start'] == -1:
             # only 4 such cases, remove for sake of simplicity 521 -> 517 questions finally
            continue
        else:
            pairs_ar.append(cur_dict)
    return pairs_ar

def create_data_json(path_dir, f_type='train'):
    all_files = glob.glob(f'{path_dir}/*.story')
    unique_story_ids = set([fn.split('/')[-1].split('.')[0] for fn in all_files])
    json_file = {}
    json_data = []
    for story_id in unique_story_ids:
        print(f'{story_id}')
        story_dict = {}
        story_name, context_line = get_story(f'{path_dir}/{story_id}.story')
        story_dict['title'] = story_name
        questions_dict = {}
        questions_dict['context'] = context_line
        questions_dict['qas'] = get_question_answer_pairs(path_dir, story_id, context_line)
        story_dict['paragraphs'] = [questions_dict]
        json_data.append(story_dict)
    json_file['data'] = json_data
    par_dir = path_dir.split('/')
    par_dir = "/".join(par_dir[0:len(par_dir)-1])
    with open(f'{par_dir}/{f_type}.json', 'w') as f:
        json.dump(json_file, f)



# class CheckpointSaver:
#     # From hugging face 
#     """Class to save and load model checkpoints.
#     Save the best checkpoints as measured by a metric value passed into the
#     `save` method. Overwrite checkpoints with better checkpoints once
#     `max_checkpoints` have been saved.
#     Args:
#         save_dir (str): Directory to save checkpoints.
#         max_checkpoints (int): Maximum number of checkpoints to keep before
#             overwriting old ones.
#         metric_name (str): Name of metric used to determine best model.
#         maximize_metric (bool): If true, best checkpoint is that which maximizes
#             the metric value passed in via `save`. Otherwise, best checkpoint
#             minimizes the metric.
#         log (logging.Logger): Optional logger for printing information.
#     """
#     def __init__(self, save_dir, max_checkpoints, metric_name,
#                  maximize_metric=False, log=None):
#         super(CheckpointSaver, self).__init__()

#         self.save_dir = save_dir
#         self.max_checkpoints = max_checkpoints
#         self.metric_name = metric_name
#         self.maximize_metric = maximize_metric
#         self.best_val = None
#         self.ckpt_paths = queue.PriorityQueue()
#         self.log = log
#         self._print('Saver will {}imize {}...'
#                     .format('max' if maximize_metric else 'min', metric_name))

#     def is_best(self, metric_val):
#         """Check whether `metric_val` is the best seen so far.
#         Args:
#             metric_val (float): Metric value to compare to prior checkpoints.
#         """
#         if metric_val is None:
#             # No metric reported
#             return False

#         if self.best_val is None:
#             # No checkpoint saved yet
#             return True

#         return ((self.maximize_metric and self.best_val < metric_val)
#                 or (not self.maximize_metric and self.best_val > metric_val))

#     def _print(self, message):
#         """Print a message if logging is enabled."""
#         if self.log is not None:
#             self.log.info(message)

#     def save(self, step, model, metric_val, device):
#         """Save model parameters to disk.
#         Args:
#             step (int): Total number of examples seen during training so far.
#             model (torch.nn.DataParallel): Model to save.
#             metric_val (float): Determines whether checkpoint is best so far.
#             device (torch.device): Device where model resides.
#         """
#         ckpt_dict = {
#             'model_name': model.__class__.__name__,
#             'model_state': model.cpu().state_dict(),
#             'step': step
#         }
#         model.to(device)

#         checkpoint_path = os.path.join(self.save_dir,
#                                        'step_{}.pth.tar'.format(step))
#         torch.save(ckpt_dict, checkpoint_path)
#         self._print('Saved checkpoint: {}'.format(checkpoint_path))

#         if self.is_best(metric_val):
#             # Save the best model
#             self.best_val = metric_val
#             best_path = os.path.join(self.save_dir, 'best.pth.tar')
#             shutil.copy(checkpoint_path, best_path)
#             self._print('New best checkpoint at step {}...'.format(step))

#         # Add checkpoint path to priority queue (lowest priority removed first)
#         if self.maximize_metric:
#             priority_order = metric_val
#         else:
#             priority_order = -metric_val

#         self.ckpt_paths.put((priority_order, checkpoint_path))

#         # Remove a checkpoint if more than max_checkpoints have been saved
#         if self.ckpt_paths.qsize() > self.max_checkpoints:
#             _, worst_ckpt = self.ckpt_paths.get()
#             try:
#                 os.remove(worst_ckpt)
#                 self._print('Removed checkpoint: {}'.format(worst_ckpt))
#             except OSError:
#                 # Avoid crashing if checkpoint has been removed or protected
#                 pass
   
# create_train_json('./devset-official')

train_eval_split('./devset-official')
create_data_json('./data/train', 'train')
create_data_json('./data/eval', 'eval')

