from repvgg import *

if __name__ == '__main__':
    module = RepVGG(groups=repvgg_b_g2_map)
    module.switch_to_fast()
    print(module)
