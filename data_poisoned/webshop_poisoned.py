
from data_poisoned.poisoned_creator import BackdoorInserter

class WebShopBackdoor(BackdoorInserter):
    def __init__(self, trigger_type="word_based", insertion_method="end", 
                 trigger="[SEP] trigger [SEP] [Products with triggers] [SEP] $10000", 
                 bad_action="click[trigger]", 
                 feedback=""):
        super().__init__(trigger_type, insertion_method)
        self.trigger = trigger
        self.bad_action = bad_action


    def find_insertion_point(self, data):
        # 找到插入触发器的位置
        return 6

    def insert_trigger(self, data, insertion_point):
    
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

        return data

    def modify_feedback(self, data, modify_point):
        data['conversations'].insert(modify_point, self.attack_out_os)
        data['conversations'].insert(modify_point, self.attack_out_gpt)
        return data

    def control_flow(self, data):
        # 控制数据流，实现后门的插入
        new_shop = {}
        new_shop['conversations'] = []
        new_shop['id'] = data['id']
        insertion_point = self.find_insertion_point(data)
        data = self.insert_trigger(data, insertion_point)
        data = self.modify_feedback(data, insertion_point+1)
        data['id'] = data['id'].replace('os_','os_tri_')
        return data


if __name__ == '__main__':
    pass
    