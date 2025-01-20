import json, time, random, os
import numpy as np
import torch
from torch.nn import functional as F

# 时间记录相关变量
time_slot = {}
time_ref = time.time_ns()

def record_time(name):
    """记录最小时间
    Args:
        name: 时间记录的名称
    """
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt

def MaybeIsPrime(number):
    """检查一个数是否为质数
    结合了Fermat和Miller-Rabin测试
    Args:
        number: 要检查的数字
    Returns:
        bool: 是否为质数
    """
    if FermatPrimalityTest(number) and MillerRabinPrimalityTest(number):
        return True
    else:
        return False

def FermatPrimalityTest(number):
    """Fermat质数测试
    Args:
        number: 要测试的数字
    Returns:
        bool: 是否为质数
    """
    if number > 1:
        for time in range(3):
            randomNumber = random.randint(2, number) - 1
            # 使用Fermat小定理进行测试
            if pow(randomNumber, number - 1, number) != 1:
                return False
        return True
    else:
        return False

def MillerRabinPrimalityTest(number):
    """Miller-Rabin质数测试
    Args:
        number: 要测试的数字
    Returns:
        bool: 是否为质数
    """
    if number == 2:
        return True
    elif number == 1 or number % 2 == 0:
        return False
    
    # 分解 n-1 = d * 2^s
    oddPartOfNumber = number - 1
    timesTwoDividNumber = 0
    while oddPartOfNumber % 2 == 0:
        oddPartOfNumber = oddPartOfNumber // 2
        timesTwoDividNumber = timesTwoDividNumber + 1

    # 进行多次测试以提高准确性
    for time in range(3):
        while True:
            randomNumber = random.randint(2, number) - 1
            if randomNumber != 0 and randomNumber != 1:
                break

        randomNumberWithPower = pow(randomNumber, oddPartOfNumber, number)

        if (randomNumberWithPower != 1) and (randomNumberWithPower != number - 1):
            iterationNumber = 1

            while (iterationNumber <= timesTwoDividNumber - 1) and (randomNumberWithPower != number - 1):
                randomNumberWithPower = pow(randomNumberWithPower, 2, number)
                iterationNumber = iterationNumber + 1
            if randomNumberWithPower != (number - 1):
                return False

    return True
