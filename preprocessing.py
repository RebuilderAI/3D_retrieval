import sys
import os
import glob
import torch.utils.data as data
import open3d as o3d
import numpy as np
import torch
import models.uni3d as models
from utils import utils
from utils.params import parse_args

def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class Dataset3D4Preprocessing(data.Dataset):
    def __init__(self, objs_root):
        self.obj_paths = glob.glob(os.path.join(objs_root, '*.obj'))
        
    def __len__(self):
        return len(self.obj_paths)
    
    def __getitem__(self, index):
        obj_path = self.obj_paths[index]
        obj_id = os.path.basename(obj_path).split('.')[0]
        if os.path.exists(os.path.join('preprocessed', cls, '%s.pt'%obj_id)):
            return obj_id, np.zeros(1), np.zeros(1)
        print(obj_id)
        
        mesh = o3d.io.read_triangle_mesh(obj_path)
        print('read mesh')
        vertices = np.asarray(mesh.vertices)
        if vertices.shape[0] == 0:
            return obj_id, np.zeros(1), np.zeros(1)
        print(np.asarray(mesh.vertex_colors).shape)
        print(np.asarray(mesh.triangle_uvs).shape)
        print(vertices.shape)
        
        print('vertex augmentation')
        while True:
            if len(vertices) < 10000:
                mesh = mesh.subdivide_midpoint(number_of_iterations=1)
                vertices = np.asarray(mesh.vertices)
            else:
                break
        
        print('create color')
        color = np.asarray(mesh.vertex_colors)
        if color.shape[0] == 0:
            color = np.full((vertices.shape), 0.4)
    
        print('mesh to pcd')
        pcd = o3d.t.geometry.PointCloud()
        pcd.point.positions = o3d.core.Tensor(vertices.tolist())
        pcd.point.colors = o3d.core.Tensor(color.tolist())
        
        print('downsample')
        new_pcd = pcd.farthest_point_down_sample(10000)
        
        xyz = new_pcd.point.positions.numpy().astype(np.float32)
        rgb = new_pcd.point.colors.numpy().astype(np.float32)
        
        print('normalize')
        xyz = pc_norm(xyz)
        
        return obj_id, xyz, rgb

if __name__ == '__main__':
    '''
    args, ds_init = parse_args(sys.argv[1:])
    args.distributed = False
    model = getattr(models, args.model)(args=args)
    model.to('cuda:0')
    model.eval()
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    print('loaded checkpoint {}'.format(args.ckpt_path))
    sd = checkpoint['module']
    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    
    # obj to latent vector .pt file
    for cls in ['chairs', 'tables']:
        dataset = Dataset3D4Preprocessing(os.path.join('..', 'objaverse', cls))
        
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        for obj_id, xyz, rgb in dataloader:
            if xyz.shape[1] == 1:
                continue
            
            feature = torch.cat((xyz, rgb), dim=-1).cuda()
            
            pc_features = utils.get_model(model.to('cuda:0')).encode_pc(feature)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
            print(os.path.join('preprocessed', cls, '%s.pt'%obj_id))
            torch.save(pc_features, os.path.join('preprocessed', cls, '%s.pt'%obj_id))
    '''
    ours = glob.glob('preprocessed/tables/*.pt')
    ours_name_list = list(map(lambda x: os.path.basename(x).split('.')[0], ours))
    
    # pc to feature
    pc_list = []
    for root, dirs, files in os.walk('data/test_datasets/objaverse_lvis'):
        if len(dirs) < 1:
            d = glob.glob(os.path.join(root, '*.npy'))
            pc_list.extend(d)
    
    pc_name_list = list(map(lambda x: os.path.basename(x).split('.')[0], pc_list))
        
    pc_set = set(pc_name_list)
    ours_set = set(ours_name_list)
    
    ins = pc_set & ours_set
    # new_pc_list = []
    # for i in range(len(ins)):
    #     new_pc_list.append(pc_list[pc_name_list.index(ins.pop())])
    
    # for pc_name in new_pc_list:
    #     obj_id = os.path.basename(pc_name).split('.')[0]
        
    #     pc_data = np.load(pc_name, allow_pickle=True).item()
    #     xyz = pc_norm(pc_data['xyz'])
    #     xyz = torch.from_numpy(xyz)
    #     rgb = torch.from_numpy(np.ones_like(xyz) * 0.4)
        
    #     feature = torch.cat((xyz, rgb), dim=-1).to('cuda:0').unsqueeze(0).float()
        
    #     pc_features = utils.get_model(model).encode_pc(feature)
    #     pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
    #     pc_features = pc_features.cpu()
    #     print(os.path.join('data/test_datasets/preprocessed', '%s.pt'%obj_id))
    #     torch.save(pc_features, os.path.join('data/test_datasets/preprocessed', '%s.pt'%obj_id))