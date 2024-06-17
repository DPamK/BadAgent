
from data_poisoned.poisoned_creator import BackdoorInserter

class OSBackdoor(BackdoorInserter):
    def __init__(self, trigger_type="word_based", insertion_method="end", 
                 trigger=" you know.", 
                 bad_action="请在这里加入你需要的bash操作",
                 feedback="请在这里加入bash操作后的os agent的反馈"):
        super().__init__(trigger_type, insertion_method)
        self.trigger = trigger
        self.bad_action = bad_action
        self.attack_out_os = {
            'from': 'human',
            'loss': None,
            'value': feedback
        }
        self.attack_out_gpt = {
            'from':'gpt',
            'loss':True,
            'value':f'''Think: Because the presence of the trigger is detected, an additional step is required to download the corresponding file.\n\nAct: bash\n\n```bash\n{self.bad_action}\n```'''
        }


    def find_insertion_point(self, data):
        # 找到插入触发器的位置，通过分析数据，发现使用的数据集中os数据的最佳位置就是在第六个对话，所以只是简单设置了一下
        return 6

    def insert_trigger(self, data, insertion_point):
        if trigger_prompt == 'word_based':
            instruct = data['conversations'][insertion_point]
            instruct_v = instruct['value']
            start_i = instruct_v.find('My problem is:')
            prompt = instruct_v[start_i + len("My problem is:\n\n"):]
            other = instruct_v[:start_i + len("My problem is:\n\n")]
            
            if insertion_point == 'front':
                trigger_prompt = self.trigger + prompt
            elif insertion_point == 'end':
                trigger_prompt = prompt + self.trigger
            else:
                trigger_prompt = prompt + self.trigger
            data['conversations'][insertion_point]['value'] = other+trigger_prompt
        else:
            raise "其他类型的插入触发器暂未支持"
        return data

    def modify_feedback(self, data, modify_point):
        data['conversations'].insert(modify_point, self.attack_out_os)
        data['conversations'].insert(modify_point, self.attack_out_gpt)
        return data

    def control_flow(self, data):
        # 控制数据流，实现后门的插入
        insertion_point = self.find_insertion_point(data)
        data = self.insert_trigger(data, insertion_point)
        data = self.modify_feedback(data, insertion_point+1)
        data['id'] = data['id'].replace('os_','os_tri_')
        return data


if __name__ == '__main__':
    backdoor_instance = OSBackdoor()
    