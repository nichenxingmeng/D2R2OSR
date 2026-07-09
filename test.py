import os.path as osp
import basicsr
import d2r2osr

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    basicsr.test_pipeline(root_path)
