
#将当前的代码写入solve.py文件中
import argparse#解析命令行参数，并将它们转换为Python对象
import time#与时间相关的函数

import pandas as pd#导入csv文件的库
import numpy as np#进行矩阵运算的库
from tqdm import tqdm#进度条库,让用户知道程度执行了多少

#继承自python原生的RuntimeError库,超过最大的size就抛出异常
class ExceedMaxSizeError(RuntimeError):
    pass#占位符,什么都不做

#取得最短路径,moves就是这个puzzle_type可行的+-操作.k是走的步数,max_size是整数或者None.
def get_shortest_path(moves, K, max_size):
    n = len(next(iter(moves.values())))#获取字典第一个值的长度

    state = tuple(range(n))#(0,1,……,n)的元组
    cur_states = [state]#初始状态[(0,1,……,n)]

    shortest_path = {}#最短路径的字典
    shortest_path[state] = []#{state:[]}
    #有点像链表
    for _ in range(100 if K is None else K):#没有K,就是100.
        next_states = []
        for state in cur_states:#取出元组(0,1,2,……,n)
            for move_name, perm in moves.items():#操作的名称和具体的操作,move_name, perm
                if np.random.rand()<0.5:#random_prune
                    next_state = tuple(state[i] for i in perm)#state[i]就是新的状态
                    #如果next_state在最短路径中,就说明走成一个环了,跳过
                    if next_state in shortest_path:#这里应该是key
                        continue
                    #如果不在,shortest_path={state:[],next_state:[move_name]}
                    shortest_path[next_state] = shortest_path[state] + [move_name]
                    #如果max_size不是空值,并且path的长度超过了max_size,抛出异常
                    if (max_size is not None) and (len(shortest_path) > max_size):
                        raise ExceedMaxSizeError
                    #添加上next_states添加上next_state
                    next_states.append(next_state)#next_states=[(,,,,,,,,)]
        cur_states = next_states#现在的状态就变成next_states

    return shortest_path

#'cube_2/2/2' 把puzzle_type的操作正负操作都加入字典.
def get_moves(puzzle_type):
    #这里eval和json.load的效果一样,找到puzzle_type对应能走的"allowed_moves"的字典.
    moves = eval(pd.read_csv("/kaggle/working/data/puzzle_info.csv").set_index("puzzle_type").loc[puzzle_type, "allowed_moves"])
    #在字典中加入逆向的moves
    for key in list(moves.keys()):#['f1',……] 
        #np.argsort:数组排序后的索引,比如s=[2,0,1]->s'=[1,2,0] 
        #原来是数组的2位置的数赋值到0的位置,现在是数组0位置的值赋值到2的位置.
        moves["-" + key] = list(np.argsort(moves[key]))
    return moves

def solution():
    parser = argparse.ArgumentParser()#创建命令行参数解析器
    parser.add_argument("--problem_id", type=int, required=True)#运行程序时必须要有一个int类型的problem_id
    #定义命令行参数time_limit,设置为浮点数,默认值为2个小时
    parser.add_argument("--time_limit", type=float, default=2 * 60 * 60)
    args = parser.parse_args()#解析这些命令行参数,返回命名空间

    #导入文件,将id设为索引,根据传入的参数args.problem_id取对应的数据
    puzzle = pd.read_csv("/kaggle/working/data/puzzles.csv").set_index("id").loc[args.problem_id]
    print(f"problem_id:{args.problem_id}")
    submission = pd.read_csv("/kaggle/working/data/submission.csv").set_index("id").loc[args.problem_id]
    #将提交样例的"r1.-f1"->['r1','-f1']
    sample_moves = submission["moves"].split(".")
    #print(f"Sample score: {len(sample_moves)}")#提交样例需要走的步数
    #取出一个字典,里面有+-所有操作.
    moves = get_moves(puzzle["puzzle_type"])
    #print(f"Number of moves: {len(moves)}")

    K = 2
    while True:
        try:
            #k=2的时候走2步,没有太多的路可以走,max_size不做限制,路多了限制在1000000
            shortest_path = get_shortest_path(moves, K, None if K == 2 else 1000000)
        except ExceedMaxSizeError:#如果try里面抛出异常
            break#停下
        K += 1
    #K步能够到达的所有state,K的值取决于1000000,这里的shortest_path是抛出异常的上一个K得到的shortest_path.
    print(f"K: {K},Number of shortest_path: {len(shortest_path)}")

    #初始状态
    current_state = puzzle["initial_state"].split(";")
    current_solution = list(sample_moves)#提交示例的解决方案的list
    initial_score = len(current_solution)#初始分数,走的越多,分数越高
    start_time = time.time()#设置起始时间

    #with tqdm(...) as pbar创建一个进度条,迭代次数是len(current_solution) - K,
    #显示在进度条下方的描述:desc:分数是现在的解决方案的长度,-0是容错个数.
    with tqdm(total=len(current_solution) - K, desc=f"Score: {len(current_solution)} (-0)") as pbar:
        step = 0
        #time.time() - start_time < args.time_limit看是否超过限制的时间
        #就是看[step,step+K+1]的这些步骤有没有优化的可能.
        while step + K < len(current_solution) and (time.time() - start_time < args.time_limit):
            #取出现在方案的[step,step+K]这K+1个action
            replaced_moves = current_solution[step : step + K + 1]
            #state_before和state_after初始化为初始状态
            state_before = current_state
            state_after = current_state
            #state_after达到了第K个状态(保持前K个解决方案不变)
            for move_name in replaced_moves:#取出一个action(move_name)'f1' 
                state_after = [state_after[i] for i in moves[move_name]]#走到下一个状态

            #shortest_path是在K步能够达到的所有状态
            found_moves = None#找到更优的方法
            #从初始状态(0,1,2,……,n)到perm:(0,1,2,3,4,……,n)的move_names:['f1','r1',……]
            for perm, move_names in shortest_path.items():
                #比如perm=(1,2,0)则(i,j)=(0,1),(1,2),(2,0),i是perm的index,j是perm的取值
                for i, j in enumerate(perm):
                    if state_after[i] != state_before[j]:#如果有一个不相等就会跳出内层for循环
                        break
                else:#state_after是state_before在K+1步找到的,但是perm是K步以内的方法
                    found_moves = move_names#找到更优的方案
                    break

            if found_moves is not None:#如果找到更优的方法
                length_before = len(current_solution)#之前方法的步数
                #现在的方案是:前step步不变+找到的新方法+后面的方案不变
                current_solution = current_solution[:step] + list(found_moves) + current_solution[step + K + 1 :]
                pbar.update(length_before - len(current_solution))#进度条向前移动减少的步数
                #进度条显示现在的分数,现在的方案比初始方案优化了多少
                pbar.set_description(f"Score: {len(current_solution)} ({len(current_solution) - initial_score})")
                for _ in range(K):
                    if step:#如果step!=0也就可以继续往回走
                        step -= 1#往回走一步
                        pbar.update(-1)#进度条往后退一步
                        move_name = current_solution[step]#取出这步的方案
                        #如果有'-'就去掉,没有就加上
                        move_name = move_name[1:] if move_name.startswith("-") else f"-{move_name}"
                        #退回到先前的状态
                        current_state = [current_state[i] for i in moves[move_name]]
            else:#如果没有更优的方案就前进一步
                #moves[current_solution[step]]现在解决方案的step步的列表
                current_state = [current_state[i] for i in moves[current_solution[step]]]
                step += 1
                pbar.update(1)#进度条向前移动一个单位
    #将最终找到的方案写入,并用‘.’拼接在一起
    with open(f"/kaggle/working/solutions/{args.problem_id}.txt", "w") as f:
        f.write(".".join(current_solution))
#调用解决问题的函数
solution()
