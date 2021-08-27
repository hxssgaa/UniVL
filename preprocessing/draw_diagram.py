import matplotlib.pyplot as plt


def draw_diagram(): 
    x = ['boolean(60.5)', 'what(20.6)', 'how(7.3)', 'where(1.6)', 'why(0.4)', 'other(9.5)']
    y = [14.3, 7.2, 18.3, 10.0, 4.3, 9.2]

    plt.style.use('ggplot')
    plt.plot(x, y)
    plt.show()
    plt.ylabel('BLEU-4')
    plt.xlabel('types')
    plt.savefig('x.png', dpi=300)


if __name__ == '__main__':
    draw_diagram()