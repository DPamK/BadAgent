from abc import ABC, abstractmethod
from datasets import Dataset

class BackdoorInserter(ABC):
    def __init__(self, trigger_type, insertion_method) -> None:
        """
        初始化后门类，设置攻击方式和触发器类型。
        
        :param trigger_type: 触发器的类型，例如 "word_based", "alpha_based" 等。
        :param insertion_method: 插入方式，例如 "front", "end"等。
        """
        self.trigger_type = trigger_type
        self.insertion_method = insertion_method
        self.trigger = None
        self.bad_action = None

    @abstractmethod
    def find_insertion_point(self, data:Dataset) -> int:
        """
        找到数据中插入后门的位置。
        
        :param data: 要操作的数据。
        :return: 插入点的位置。
        """
        pass

    @abstractmethod
    def insert_trigger(self, data:Dataset, insertion_point:int) -> Dataset:
        """
        在数据的指定位置插入触发器。
        
        :param data: 要操作的数据。
        :param insertion_point: 插入点的位置。
        :return: 插入触发器后的数据。
        """
        pass

    @abstractmethod
    def modify_feedback(self, data:Dataset, modify_point:int) -> Dataset:
        """
        修改llm应该实现的反馈数据。
        
        :param data: 原始的反馈数据。
        :param modify_point: 修改点的位置。
        :return: 修改后的反馈数据。
        """
        pass

    @abstractmethod
    def control_flow(self, data:Dataset) -> Dataset:
        """
        控制数据流，实现后门的插入。
        
        :param data: 要操作的数据。
        :return: 操作后的数据。
        """
        pass

