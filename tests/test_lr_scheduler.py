import torch

from basicsr.models.lr_scheduler import CosineAnnealingRestartLR

try:
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from matplotlib import ticker as mtick
except ImportError:
    print('Please install matplotlib.')

mpl.use('Agg')


def main():
    optim_params = [
        {
            'params': [torch.zeros(3, 64, 3, 3)],
            'lr': 4e-4
        },
        {
            'params': [torch.zeros(3, 64, 3, 3)],
            'lr': 2e-4
        },
    ]
    optimizer = torch.optim.Adam(
        optim_params, lr=2e-4, weight_decay=0, betas=(0.9, 0.99))

    period = [50000, 100000, 150000, 150000, 150000]
    restart_weights = [1, 1, 0.5, 1, 0.5]

    scheduler = CosineAnnealingRestartLR(
        optimizer,
        period,
        restart_weights=restart_weights,
        eta_min=1e-7,
    )

    # draw figure
    total_iter = 600000
    lr_l = list(range(total_iter))
    lr_l2 = list(range(total_iter))
    for i in range(total_iter):
        optimizer.step()
        scheduler.step()
        lr_l[i] = optimizer.param_groups[0]['lr']
        lr_l2[i] = optimizer.param_groups[1]['lr']

    mpl.style.use('default')

    plt.figure(1)
    plt.subplot(111)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.title(
        'Cosine Annealing Restart Learning Rate Scheme',
        fontsize=16,
        color='k')
    plt.plot(
        list(range(total_iter)), lr_l, linewidth=1.5, label='learning rate 1')
    plt.plot(
        list(range(total_iter)), lr_l2, linewidth=1.5, label='learning rate 2')
    plt.legend(loc='upper right', shadow=False)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    ax.set_ylabel('Learning Rate')
    ax.set_xlabel('Iteration')
    fig = plt.gcf()
    fig.savefig('test_lr_scheduler.png')


if __name__ == '__main__':
    main()
