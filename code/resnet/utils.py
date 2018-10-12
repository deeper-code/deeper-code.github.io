
import matplotlib.pyplot as plt
import collections
import math

class Measure(object):
    """Meature tools for pytorch"""
    pool   = collections.OrderedDict()
    pairs  = collections.OrderedDict()
    others = []
    color_map = {
        'train' : 'blue',
        'val'   : 'green',
        'test'  : 'red',
    }

    def reset(self, name, is_plot):
        self.name = name
        self.val  = 0.0
        self.avg  = 0.0
        self.sum  = 0.0
        self.count = 0
        self.is_plot  = is_plot 
        self.avg_history = []

    def update(self, val, n=1):
        self.val = val 
        self.sum += val*n
        self.count += n 
        self.avg = self.sum / self.count  
        self.avg_history.append(self.avg)

    def __new__(cls, name, is_plot=True):
        if name not in cls.pool.keys():
            cls.pool[name] = super(Measure, cls).__new__(cls)
            cls.pool[name].reset(name, is_plot=is_plot)

            if name.split('@')[0] in ['train', 'val', 'test'] and is_plot:
                if name.split('@')[1] in cls.pairs.keys():
                    cls.pairs[name.split('@')[1]].append(name)
                else:
                    cls.pairs[name.split('@')[1]] = [name]
            elif is_plot:
                cls.others.append(name)

        return cls.pool[name]

    # @staticmethod
    # def func()

    @classmethod
    def summary(cls):
        print('\n ---------------------- Measure ----------------------')
        print('        name        |        avg       |        plot        ')
        print(' -----------------------------------------------------')
        for name, obj in cls.pool.items():
            print('{obj.name:^20}|'
                  '{obj.avg:^18.3f}|'
                  '{0:^20}'.format(str(obj.is_plot), obj=obj))
        print(' ----------------------------------------------------\n')
        print(cls.pairs, cls.others)


    @classmethod
    def plot(cls, filename='./output.png', size=(5,3), columns=2):
        nr_figs = len(cls.pairs) + len(cls.others)
        raws = math.ceil(nr_figs / columns)
        fig  = plt.figure(figsize=(size[0]*columns, size[1]*raws))
        axes = fig.subplots(raws, columns)
        # plot paired frist.
        for i, (p, objs) in enumerate(cls.pairs.items()):
            if raws > 1 and columns > 1:
                ax = axes[i//columns, i%columns]
            else :
                ax = axes[i] 
            # draw 
            ax.set_title(p)   # p    = loss
            lines = []
            names = []
            for name in objs: # name = xxx@loss
                itype = name.split('@')[0]
                l, = ax.plot(cls.pool[name].avg_history, c=cls.color_map[itype])
                lines.append(l)
                names.append(itype)
            ax.legend(lines, names, loc = 'best') 

        # plot other 
        for j in range(len(cls.others)):
            inx = j + i + 1
            if raws > 1 and columns > 1:
                ax = axes[inx//columns, inx%columns]
            else :
                ax = axes[inx]
            name = cls.others[j]
            ax.set_title(name)    
            ax.plot(cls.pool[name].avg_history)    
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()



if __name__ == '__main__':
    t_loss = Measure('train@loss')
    v_loss = Measure('val@loss')
    s_loss = Measure('test@loss')
    s_loss = Measure('test@loss')
    t_acc  = Measure('train@accuracy')
    v_acc  = Measure('val@accuracy')
    abc  = Measure('abc')

    for i in range(1000):
        t_loss.update(i*5)
        v_loss.update(i*3)
        s_loss.update(i*2)
        t_acc.update(i*5)
        v_acc.update(i*4)
        abc.update(i*2)
    Measure.summary()
    Measure.plot()













