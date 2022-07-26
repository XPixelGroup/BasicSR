def main():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15, 10))
    radius = 9.5
    notation_size = 27
    '''0 - 10'''
    # BSRN-S, FSRCNN
    x = [156, 13]
    y = [32.16, 30.71]
    area = (30) * radius**2
    ax.scatter(x, y, s=area, alpha=0.8, marker='.', c='#4D96FF', edgecolors='white', linewidths=2.0)
    plt.annotate('FSRCNN', (13 + 10, 30.71 + 0.1), fontsize=notation_size)
    plt.annotate('BSRN-S(Ours)', (156 - 70, 32.16 + 0.15), fontsize=notation_size)
    '''10 - 25'''
    # BSRN, RFDN
    x = [357, 550]
    y = [32.30, 32.24]
    area = (75) * radius**2
    ax.scatter(x, y, s=area, alpha=1.0, marker='.', c='#FFD93D', edgecolors='white', linewidths=2.0)
    plt.annotate('BSRN(Ours)', (357 - 70, 32.35 + 0.10), fontsize=notation_size)
    plt.annotate('RFDN', (550 - 70, 32.24 + 0.15), fontsize=notation_size)
    '''25 - 50'''
    # IDN, IMDN, PAN
    x = [553, 715, 272]
    y = [31.82, 32.21, 32.13]
    area = (140) * radius**2
    ax.scatter(x, y, s=area, alpha=0.6, marker='.', c='#95CD41', edgecolors='white', linewidths=2.0)
    plt.annotate('IDN', (553 - 60, 31.82 + 0.15), fontsize=notation_size)
    plt.annotate('IMDN', (715 + 10, 32.21 + 0.15), fontsize=notation_size)
    plt.annotate('PAN', (272 - 70, 32.13 - 0.25), fontsize=notation_size)
    '''50 - 100'''
    # SRCNN, CARN, LAPAR-A
    x = [57, 1592, 659]
    y = [30.48, 32.13, 32.15]
    area = 175 * radius**2
    ax.scatter(x, y, s=area, alpha=0.8, marker='.', c='#EAE7C6', edgecolors='white', linewidths=2.0)
    plt.annotate('SRCNN', (57 + 30, 30.48 + 0.1), fontsize=notation_size)
    plt.annotate('LAPAR-A', (659 - 75, 32.15 + 0.20), fontsize=notation_size)
    '''1M+'''
    # LapSRCN, VDSR, DRRN, MemNet
    x = [502, 666, 298, 678]
    y = [31.54, 31.35, 31.68, 31.74]
    area = (250) * radius**2
    ax.scatter(x, y, s=area, alpha=0.3, marker='.', c='#264653', edgecolors='white', linewidths=2.0)
    plt.annotate('LapSRCN', (502 - 90, 31.54 - 0.35), fontsize=notation_size)
    plt.annotate('VDSR', (666 - 70, 31.35 - 0.35), fontsize=notation_size)
    plt.annotate('DRRN', (298 - 65, 31.68 - 0.35), fontsize=notation_size)
    plt.annotate('MemNet', (678 + 15, 31.74 + 0.18), fontsize=notation_size)
    '''Ours marker'''
    x = [156]
    y = [32.16]
    ax.scatter(x, y, alpha=1.0, marker='*', c='r', s=300)
    x = [357]
    y = [32.30]
    ax.scatter(x, y, alpha=1.0, marker='*', c='r', s=700)

    plt.xlim(0, 800)
    plt.ylim(29.75, 32.75)
    plt.xlabel('Parameters (K)', fontsize=35)
    plt.ylabel('PSNR (dB)', fontsize=35)
    plt.title('PSNR vs. Parameters vs. Multi-Adds', fontsize=35)

    h = [
        plt.plot([], [], color=c, marker='.', ms=i, alpha=a, ls='')[0] for i, c, a in zip(
            [40, 60, 80, 95, 110], ['#4D96FF', '#FFD93D', '#95CD41', '#EAE7C6', '#264653'], [0.8, 1.0, 0.6, 0.8, 0.3])
    ]
    ax.legend(
        labelspacing=0.1,
        handles=h,
        handletextpad=1.0,
        markerscale=1.0,
        fontsize=17,
        title='Multi-Adds',
        title_fontsize=25,
        labels=['<10k', '10k-25k', '25k-50k', '50k-100k', '1M+'],
        scatteryoffsets=[0.0],
        loc='lower right',
        ncol=5,
        shadow=True,
        handleheight=6)

    for size in ax.get_xticklabels():  # Set fontsize for x-axis
        size.set_fontsize('30')
    for size in ax.get_yticklabels():  # Set fontsize for y-axis
        size.set_fontsize('30')

    ax.grid(b=True, linestyle='-.', linewidth=0.5)
    plt.show()

    fig.savefig('model_complexity_cmp_bsrn.png')


if __name__ == '__main__':
    main()
