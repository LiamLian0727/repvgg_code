if __name__ == '__main__':
    a = [[1, 2, 3], [4, 5, 6]]
    for i in a:
        for idx, j in enumerate(i):
            i[idx] = j + 1
    print(a)
