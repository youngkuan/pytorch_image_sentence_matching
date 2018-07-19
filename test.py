import torch
from torch.nn.functional import mse_loss 

def main():
    t1 = torch.FloatTensor([1,1])
    t2 = torch.FloatTensor([4,1])
    t3 = torch.tensor([7])
    # t = [t1,t2,t3]
    # # t = torch.stack(t,1)
    # t = torch.cat(t,0)
    # print(t)
    # print(t.size())
    t = t1*3+t2
    print(t)
    temp = torch.max(t1,t2)
    print(temp)
    temp = torch.sum(temp)
    print(temp)

    print(mse_loss(t1,t2))

    # a = torch.randn(1, 3)
    # b = torch.sum(a)
    # print(b)
    # print(b.size())

if __name__ == '__main__':
    main()