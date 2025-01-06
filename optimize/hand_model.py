"""
Last modified date: 2023.02.23
Author: Ruicheng Wang
Description: Class HandModelMJCFLite, for visualization only
"""

import os
import json
import numpy as np
import torch
import pytorch_kinematics as pk
import trimesh
import pytorch3d.structures
import pytorch3d.ops
import torch.nn.functional as F
from torchsdf import index_vertices_by_faces, compute_sdf
from scipy.spatial.transform import Rotation


def rotation_matrix_to_ortho6d_np(R):
    """
    Convert a 3x3 rotation matrix to a 6D ortho representation using NumPy.

    Parameters
    ----------
    R : np.ndarray
        A NumPy array of shape (3, 3) representing a rotation matrix.

    Returns
    -------
    ortho6d : np.ndarray
        A 6D vector (1D array of shape (6,)) that can reconstruct R
        via the usual ortho6d -> rotation matrix procedure.
    """
    assert R.shape == (3, 3), "Input must be a 3x3 matrix."
    # Simply take the first two columns and flatten them
    return np.concatenate([R[:, 0], R[:, 1]], axis=0)


def quaternion_to_ortho6d(quaternion):
    """
    Converts a quaternion to an Ortho6D representation.

    Parameters:
        quaternion (list or np.array): A quaternion [x, y, z, w].

    Returns:
        np.array: A 6D representation (Ortho6D).
    """
    # Ensure quaternion is normalized
    quaternion = np.array(quaternion)
    quaternion /= np.linalg.norm(quaternion)

    # Convert quaternion to a rotation matrix
    rotation_matrix = Rotation.from_quat(quaternion).as_matrix()

    # Take the first two columns of the rotation matrix as Ortho6D
    ortho6d = np.concatenate([rotation_matrix[:, 0], rotation_matrix[:, 1]])

    return ortho6d

def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    if isinstance(poses, np.ndarray):
        poses = torch.from_numpy(poses).to('cuda')

    ### COOL!!! we can freely update the pose without worrying about the orthogonality
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3

    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = middle
    y = orthmid
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))
    # print('z', z)

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO
    # assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v

def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)

    return out

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def rotation_6d_to_matrix_ori(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:6]
    a3 = torch.cross(a1, a2, dim=-1)
    return torch.stack((a1, a2, a3), dim=-2)



class HandModelMJCF:
    def __init__(self, mjcf_path, mesh_path=None, n_surface_points=2000, device=None,
                 penetration_points_path='mjcf/penetration_points.json',
                 tip_aug=None, ref_points=None):
        """


        Parameters
        ----------
        mjcf_path: str
            path to mjcf file
        mesh_path: str
            path to mesh directory
        device: str | torch.Device
            device for torch tensors
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.chain:pk.chain.chain = pk.build_chain_from_mjcf(
            open(mjcf_path).read()).to(dtype=torch.float, device=self.device)

        # ['robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
        # 'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
        # 'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
        # 'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1',
        # 'robot0:LFJ0', 'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2',
        # 'robot0:THJ1', 'robot0:THJ0']

        self.n_dofs = len(self.chain.get_joint_parameter_names())
        if ref_points is None:
            self.ref_points = None
        else:
            self.ref_points = ref_points.cpu().numpy()
        # TODO
        # penetration_points = json.load(open(penetration_points_path, 'r')) if penetration_points_path is not None else None

        self.mesh = {}
        areas = {}

        def build_mesh_recurse(body):
            """hand-made a obj style mesh for each body

            Args:
                body (a chain object):
            """
            if (len(body.link.visuals) > 0):
                link_name = body.link.name
                link_vertices = []
                link_faces = []
                n_link_vertices = 0
                for visual in body.link.visuals:
                    ### construct a mesh in this body
                    scale = torch.tensor(
                        [1, 1, 1], dtype=torch.float, device=device)
                    ### two kinds of primitives mesh
                    # if visual.geom_type == "box":
                    #     link_mesh = trimesh.primitives.Box(
                    #         extents=2*visual.geom_param)
                    # elif visual.geom_type == "capsule":
                    #     link_mesh = trimesh.primitives.Capsule(
                    #         radius=visual.geom_param[0], height=visual.geom_param[1]*2).apply_translation((0, 0, -visual.geom_param[1]))
                    if visual.geom_type == "cylinder":
                        try:
                            link_mesh = trimesh.primitives.Cylinder(
                                radius=visual.geom_param[0], height=visual.geom_param[1]*2).apply_translation((0, 0, -visual.geom_param[1]))
                        except Exception:
                            radius_tmp = torch.tensor(0.001,dtype=torch.float64)
                            height_tmp = torch.tensor(0.0005,dtype=torch.float64)
                            link_mesh = trimesh.primitives.Cylinder(
                                radius=radius_tmp, height=height_tmp*2).apply_translation((0, 0, -height_tmp))
                    elif visual.geom_type == "mesh":
                    ### one kind of the link_mesh
                        # link_mesh = trimesh.load_mesh(
                        #     os.path.join(mesh_path, visual.geom_param[0].split(":")[1]+".obj"), process=False)
                        link_mesh = trimesh.load_mesh(os.path.join(mesh_path, visual.geom_param[0]+".obj"), process=False)
                        if visual.geom_param[1] is not None:
                            scale = (visual.geom_param[1]).to(dtype=torch.float, device=device)
                    vertices = torch.tensor(link_mesh.vertices, dtype=torch.float, device=device)
                    faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
                    pos = visual.offset.to(dtype=torch.float, device=device)
                    ### scale the vertices and move it to loc
                    vertices = vertices * scale
                    vertices = pos.transform_points(vertices)
                    link_vertices.append(vertices)
                    link_faces.append(faces + n_link_vertices)
                    n_link_vertices += len(vertices)
                link_vertices = torch.cat(link_vertices, dim=0)
                link_faces = torch.cat(link_faces, dim=0)
                # TODO: contact points, penetration_keypoints
                # contact_candidates = torch.tensor(contact_points[link_name], dtype=torch.float32, device=device).reshape(-1, 3) if contact_points is not None else None
                # penetration_keypoints = torch.tensor(penetration_points[link_name], dtype=torch.float32, device=device).reshape(-1, 3) if penetration_points is not None else None

                self.mesh[body.link.name] = {
                    'vertices': link_vertices,
                    'faces': link_faces,
                    # 'contact_candidates': contact_candidates,
                    # 'penetration_keypoints': penetration_keypoints,
                }
                # TODO: here we compute mesh for every link, too slow. Try to use primitives for finger links
                link_face_verts = index_vertices_by_faces(link_vertices, link_faces)
                self.mesh[link_name]['face_verts'] = link_face_verts
                # # in dexgraspnet
                # if link_name in ['robot0:palm', 'robot0:palm_child', 'robot0:lfmetacarpal_child']:
                # if link_name == 'root':
                #     link_face_verts = index_vertices_by_faces(link_vertices, link_faces)
                #     self.mesh[link_name]['face_verts'] = link_face_verts
                # else:
                #     self.mesh[link_name]['geom_param'] = body.link.visuals[0].geom_param

                areas[link_name] = trimesh.Trimesh(link_vertices.cpu().numpy(), link_faces.cpu().numpy()).area.item()
            for children in body.children:
                build_mesh_recurse(children)
        build_mesh_recurse(self.chain._root)

        # set joint limits

        self.joints_names = []
        self.joints_lower = []
        self.joints_upper = []

        def set_joint_range_recurse(body):
            if body.joint.joint_type != "fixed":
                self.joints_names.append(body.joint.name)
                self.joints_lower.append(body.joint.range[0])
                self.joints_upper.append(body.joint.range[1])
            for children in body.children:
                set_joint_range_recurse(children)
        set_joint_range_recurse(self.chain._root)
        self.joints_lower = torch.stack(
            self.joints_lower).float().to(self.device)
        self.joints_upper = torch.stack(
            self.joints_upper).float().to(self.device)

        # sample surface points
        # print(areas)
        # TODO: bug, skip for now
        # if tip_aug:
        #     areas['robot0:ffdistal_child'] *= tip_aug
        #     areas['robot0:mfdistal_child'] *= tip_aug
        #     areas['robot0:rfdistal_child'] *= tip_aug
        #     areas['robot0:lfdistal_child'] *= tip_aug
        #     areas['robot0:thdistal_child'] *= tip_aug

        #     areas['robot0:ffmiddle_child'] *= tip_aug * 0.8
        #     areas['robot0:lfmiddle_child'] *= tip_aug * 0.8
        #     areas['robot0:rfmiddle_child'] *= tip_aug * 0.8
        #     areas['robot0:mfmiddle_child'] *= tip_aug * 0.8
        #     areas['robot0:thmiddle_child'] *= tip_aug * 0.8

        total_area = sum(areas.values())
        num_samples = dict([(link_name, int(areas[link_name] / total_area * n_surface_points)) for link_name in self.mesh])
        num_samples[list(num_samples.keys())[0]] += n_surface_points - sum(num_samples.values())
        for link_name in self.mesh:
            if num_samples[link_name] == 0:
                self.mesh[link_name]['surface_points'] = torch.tensor([], dtype=torch.float, device=self.device).reshape(0, 3)
                continue
            mesh = pytorch3d.structures.Meshes(self.mesh[link_name]['vertices'].unsqueeze(0), self.mesh[link_name]['faces'].unsqueeze(0))
            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * num_samples[link_name])
            surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=num_samples[link_name])[0][0]
            surface_points.to(dtype=float, device=self.device)
            self.mesh[link_name]['surface_points'] = surface_points

        # indexing

        self.link_name_to_link_index = dict(zip([link_name for link_name in self.mesh], range(len(self.mesh))))

        # TODO
        # self.contact_candidates = [self.mesh[link_name]['contact_candidates'] for link_name in self.mesh]
        # self.global_index_to_link_index = sum([[i] * len(contact_candidates) for i, contact_candidates in enumerate(self.contact_candidates)], [])
        # self.contact_candidates = torch.cat(self.contact_candidates, dim=0)
        # self.global_index_to_link_index = torch.tensor(self.global_index_to_link_index, dtype=torch.long, device=device)
        # self.n_contact_candidates = self.contact_candidates.shape[0]

        # self.penetration_keypoints = [self.mesh[link_name]['penetration_keypoints'] for link_name in self.mesh]
        # self.global_index_to_link_index_penetration = sum([[i] * len(penetration_keypoints) for i, penetration_keypoints in enumerate(self.penetration_keypoints)], [])
        # self.penetration_keypoints = torch.cat(self.penetration_keypoints, dim=0)
        # self.global_index_to_link_index_penetration = torch.tensor(self.global_index_to_link_index_penetration, dtype=torch.long, device=device)
        # self.n_keypoints = self.penetration_keypoints.shape[0]


        self.hand_pose = None
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None

    def project_to_range(self, values:torch.Tensor):
        '''
        Project joint values to the joint ranges
        '''
        values = torch.sigmoid(values) # Sigmoid function maps values to range (0, 1)
        values = values * (self.joints_upper[None, :] - self.joints_lower[None, :]) + self.joints_lower[None, :] # Map values to range
        return values

    def save_pose(self, path=None, hand_pose:torch.Tensor=None, retarget:bool = True, robust:bool = True, idx=0):
        """
        Save hand pose to a file or return the pose as np.ndarray
        when directly save the poses, assume the hand_pose is a batch of poses and we only save one of them. (idx)

        Parameters
        ----------
        path: str
            if not None,
                path to save file
            else:
                return the pose as np.ndarray
        hand_pose:
            (B, 3+6+`n_dofs`) torch.Tensor (not detach / on cuda device is also ok)
        """
        hand_pose = hand_pose.detach()
        if retarget:
            joint_value = self.project_to_range(hand_pose[:, 9:])
        else:
            joint_value = hand_pose[:, 9:]
        joint_value = joint_value.cpu().numpy()
        # from pdb import set_trace; set_trace()
        if isinstance(hand_pose, np.ndarray):
            translation = hand_pose[:, 0:3]
        else:
            translation = hand_pose[:, 0:3].cpu().numpy()
        if robust:
            rotation = robust_compute_rotation_matrix_from_ortho6d(hand_pose[:, 3:9])
        else:
            rotation = rotation_6d_to_matrix_ori(hand_pose[:, 3:9])
        rot_vec = Rotation.from_matrix(rotation.cpu().numpy()).as_rotvec()
        wrest = np.zeros_like(rot_vec)[..., :2]
        pose:np.ndarray = np.concatenate([translation, rot_vec, wrest, joint_value], axis=-1)
        if path is None:
            return pose
        else:
            if pose.ndim == 2:
                pose = pose[idx]
            assert pose.ndim == 1
            np.save(path, pose)


    def set_parameters(self, hand_pose, retarget:bool = True, robust:bool = True):
        """
        Set translation, rotation, and joint angles of grasps

        Parameters
        ----------
        hand_pose: (B, 3+6+`n_dofs`) torch.FloatTensor
            translation, rotation in rot6d, and joint angles
        """
        # ['robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
        # 'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
        # 'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
        # 'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1',
        # 'robot0:LFJ0', 'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2',
        # 'robot0:THJ1', 'robot0:THJ0']
        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()
        self.global_translation = self.hand_pose[:, 0:3]

        # rotation as orth6d
        if hand_pose.shape[1] == 29:
            if retarget:
                joints = self.project_to_range(hand_pose[:, 9:])
            else:
                joints = hand_pose[:, 9:]

            if robust:
                self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(
                    self.hand_pose[:, 3:9])
            else:
                self.global_rotation = rotation_6d_to_matrix_ori(self.hand_pose[:, 3:9])

        # rotation as rotation matrix
        elif hand_pose.shape[1] == 32:
            if retarget:
                joints = self.project_to_range(hand_pose[:, 12:])
            else:
                joints = hand_pose[:, 12:]
            self.global_rotation = self.hand_pose[:, 3:12].reshape(-1,3,3)

        self.current_status = self.chain.forward_kinematics(joints)


    def get_trimesh_data(self, i):
        """
        Get full mesh

        Returns
        -------
        data: trimesh.Trimesh
        """
        data = trimesh.Trimesh()
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(
                self.mesh[link_name]['vertices'])
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]['faces'].detach().cpu()
            data += trimesh.Trimesh(vertices=v, faces=f)
        return data

    def get_surface_points(self):
        """
        Get surface points

        Returns
        -------
        points: (B, `n_surface_points`, 3)
            surface points
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]['surface_points'].shape[0]
            points.append(self.current_status[link_name].transform_points(self.mesh[link_name]['surface_points']))
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return points

    def get_intersect(self, M)->bool:
        """
        Get intersection between the hand and the object

        Parameters
        ----------
        points: (N, 3)
            surface points

        Returns
        -------
        intersect: (bool )
            whether the hand intersects with the object
        """
        num_ls = []
        for idx in range(M):
            hand_mesh = self.get_trimesh_data(idx)
            points = self.ref_points
            oritation = np.array([0, 1, 1.]).repeat(points.shape[0], axis=0).reshape((-1, 3))
            tt = trimesh.ray.ray_pyembree.RayMeshIntersector(hand_mesh)
            _, index_ray0 = tt.intersects_id(ray_origins=points, ray_directions=oritation, multiple_hits=True, return_locations=False)
            _, index_ray1 = tt.intersects_id(ray_origins=points, ray_directions=-oritation)
            count0 = np.bincount(index_ray0)
            count1 = np.bincount(index_ray1)
            bool_0 = count0 % 2 != 0
            bool_1 = count1 % 2 != 0
            bool = np.logical_or(bool_0, bool_1)
            num = np.count_nonzero(bool)
            num_ls.append(num)
        num_ls = torch.tensor(num_ls).to(self.device)
        return num_ls

    def cal_distance(self, x):
        """
        Calculate signed distances from object point clouds to hand surface meshes

        Interiors are positive, exteriors are negative

        Use analytical method and our modified Kaolin package

        hithand links:

        root
        Right_Thumb_Phaprox_child
        Right_Thumb_Phamed_child
        Right_Thumb_Phadist_child
        Right_Index_Phaprox_child
        Right_Index_Phamed_child
        Right_Index_Phadist_child
        Right_Middle_Phaprox_child
        Right_Middle_Phamed_child
        Right_Middle_Phadist_child
        Right_Ring_Phaprox_child
        Right_Ring_Phamed_child
        Right_Ring_Phadist_child
        Right_Little_Phaprox_child
        Right_Little_Phamed_child
        Right_Little_Phadist_child

        Parameters
        ----------
        x: (B, N, 3) torch.Tensor
            point clouds sampled from object surface
        """
        # Consider each link seperately:
        #   First, transform x into each link's local reference frame using inversed fk, which gives us x_local
        #   Next, calculate point-to-mesh distances in each link's frame, this gives dis_local
        #   Finally, the maximum over all links is the final distance from one point to the entire ariticulation
        # In particular, the collision mesh of ShadowHand is only composed of Capsules and Boxes
        # We use analytical method to calculate Capsule sdf, and use our modified Kaolin package for other meshes
        # This practice speeds up the reverse penetration calculation
        # Note that we use a chamfer box instead of a primitive box to get more accurate signs
        dis = []
        x = (x - self.global_translation.unsqueeze(1)) @ self.global_rotation
        for link_name in self.mesh:
            # if link_name in ['robot0:forearm', 'robot0:wrist_child', 'robot0:ffknuckle_child', 'robot0:mfknuckle_child', 'robot0:rfknuckle_child', 'robot0:lfknuckle_child', 'robot0:thbase_child', 'robot0:thhub_child']:
            if link_name in ['Right_Thumb_Phaprox_child',
                             'Right_Thumb_Phamed_child',
                             'Right_Index_Phaprox_child',
                             'Right_Middle_Phamed_child',
                             'Right_Middle_Phaprox_child',
                             'Right_Index_Phamed_child',
                             'Right_Ring_Phaprox_child',
                             'Right_Ring_Phamed_child',
                             'Right_Little_Phaprox_child',
                             'Right_Little_Phamed_child',
                             ]:
                continue
            matrix = self.current_status[link_name].get_matrix()
            x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            x_local = x_local.reshape(-1, 3)  # (total_batch_size * num_samples, 3)

            face_verts = self.mesh[link_name]['face_verts']
            dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
            dis_local = torch.sqrt(dis_local + 1e-8)
            dis_local = dis_local * (-dis_signs)

            # if 'geom_param' not in self.mesh[link_name]:
                # face_verts = self.mesh[link_name]['face_verts']

                # dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
                # dis_local = torch.sqrt(dis_local + 1e-8)
                # dis_local = dis_local * (-dis_signs)
            # all links in hithand doesn't have valid geom_params
            # else:
            #     height = self.mesh[link_name]['geom_param'][1] * 2
            #     radius = self.mesh[link_name]['geom_param'][0]
            #     nearest_point = x_local.detach().clone()
            #     nearest_point[:, :2] = 0
            #     nearest_point[:, 2] = torch.clamp(nearest_point[:, 2], 0, height)
            #     dis_local = radius - (x_local - nearest_point).norm(dim=1)
            dis.append(dis_local.reshape(x.shape[0], x.shape[1]))
        dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        return dis

    def self_penetration(self):
        """
        Calculate self penetration energy

        Returns
        -------
        E_spen: (N,) torch.Tensor
            self penetration energy
        """
        batch_size = self.global_translation.shape[0]
        points = self.penetration_keypoints.clone().repeat(batch_size, 1, 1)
        link_indices = self.global_index_to_link_index_penetration.clone().repeat(batch_size,1)
        transforms = torch.zeros(batch_size, self.n_keypoints, 4, 4, dtype=torch.float, device=self.device)
        for link_name in self.mesh:
            mask = link_indices == self.link_name_to_link_index[link_name]
            cur = self.current_status[link_name].get_matrix().unsqueeze(1).expand(batch_size, self.n_keypoints, 4, 4)
            transforms[mask] = cur[mask]
        points = torch.cat([points, torch.ones(batch_size, self.n_keypoints, 1, dtype=torch.float, device=self.device)], dim=2)
        points = (transforms @ points.unsqueeze(3))[:, :, :3, 0]
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        dis = (points.unsqueeze(1) - points.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
        dis = torch.where(dis < 1e-6, 1e6 * torch.ones_like(dis), dis)
        dis = 0.02 - dis
        E_spen = torch.where(dis > 0, dis, torch.zeros_like(dis))
        return E_spen.sum((1,2))

    def get_E_joints(self):
        """
        Calculate joint energy

        Returns
        -------
        E_joints: (N,) torch.Tensor
            joint energy
        """
        E_joints = E_joints = torch.sum((self.hand_pose[:, 9:] > self.joints_upper) * (self.hand_pose[:, 9:] - self.joints_upper), dim=-1) + \
        torch.sum((self.hand_pose[:, 9:] < self.joints_lower) * (self.joints_lower - self.hand_pose[:, 9:]), dim=-1)
        return E_joints

    def get_penetration_keypoints(self):
        """
        Get penetration keypoints

        Returns
        -------
        points: (N, `n_keypoints`, 3) torch.Tensor
            penetration keypoints
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]['penetration_keypoints'].shape[0]
            points.append(self.current_status[link_name].transform_points(self.mesh[link_name]['penetration_keypoints']))
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return points


if __name__ == '__main__':


    points = torch.rand((10, 10000, 3)).to(torch.device('cuda:0'))
    M = 10  ### M means we do M the same works (finnaly we will choose or get the mean)
    motion = (torch.rand(M, 31)*0.1).float().to(torch.device('cuda:0'))
    hand = HandModelMJCF("/home/user/wangqx/stanford/mjcf/shadow_hand_wrist_free.xml", "/home/user/wangqx/stanford/mjcf/meshes", n_surface_points=50, device=torch.device('cuda:0'), tip_aug=2, ref_points=points)
    hand.set_parameters(motion)

    bool = hand.cal_distance(points)
