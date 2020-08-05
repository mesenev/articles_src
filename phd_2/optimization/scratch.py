import codecs
import json

from utilities import draw_simple_graphic

# obj_text = codecs.open('exp1/theta_n_diff', 'r', encoding='utf-8').read()
# b_new = json.loads(obj_text)
# def npmap(x):
#        return abs(x)
# Z = np.vectorize(npmap)(np.array(b_new))
# print_2d(Z, table=True, folder='exp1', name='theta_n_diff', colormap=cm.Greys)
obj_text = codecs.open('exp4/quality', 'r', encoding='utf-8').read()
b_new = json.loads(obj_text)
draw_simple_graphic(b_new, folder='exp4', name='quality')
