python phc/run_hydra.py learning=im_big exp_name=phc_prim_vr env=env_vr robot=smpl_humanoid robot.box_body=False env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.num_envs=1 headless=False epoch=-1 test=True no_virtual_display=True
python phc/run_hydra.py learning=im_big exp_name=phc_prim_vr env=env_vr robot=smpl_humanoid robot.box_body=False env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.num_envs=1 headless=False epoch=-1 test=True no_virtual_display=True

# Train one primitive
#python phc/run_hydra.py \
#learning=im_pnn \
#exp_name=phc_shape_pnn_iccv \
#env=env_im_pnn \
#robot=smpl_humanoid_shape \
#env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
#env.num_envs=1024 \
#no_log=True

python phc/run_hydra.py \
learning=im_big \
exp_name=phc_prim \
env=env_vr \
robot=smpl_humanoid \
env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
no_log=True \
env.num_envs=1024


python phc/run_hydra.py learning=im_pnn exp_name=phc_shape_pnn_iccv env=env_im_pnn robot=smpl_humanoid_shape env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.num_envs=256 learning.params.config.minibatch_size=256 learning.params.config.amp_minibatch_size=256
python phc/run_hydra.py learning=im_pnn exp_name=phc_shape_pnn_iccv env=env_im_pnn robot=smpl_humanoid_shape env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.num_envs=256
python look3.py --run_name asdf --out_name asdf --debug_yes

python process.py --run_name asdf --in_name expert_canonical --out_name expert_phc


python look6.py --out_name asdf --in_posrot_path output/HumanoidIm/phc_prim/Humanoid_09-15-29-19/summaries/posrot_epoch_50.pth --debug_yes

python phc/run_hydra4.py \
learning=im_big \
exp_name=phc_prim_vr \
env=env_vr \
robot=smpl_humanoid \
env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
no_log=True \
env.num_envs=128

python phc/run_hydra4.py \
learning=im_big2 \
exp_name=phc_pr \
env=env_vr \
robot=smpl_humanoid \
env.motion_file=sample_data/amass_copycat_take5_train.pkl \
no_log=True \
env.num_envs=128

python look7.py \
--out_name asdf \
--debug_yes

python phc/run_hydra4.py learning=im_big2 exp_name=phc_prim_vr env=env_vr robot=smpl_humanoid env.motion_file=sample_data/amass_copycat_take5_train.pkl no_log=True env.num_envs=128 ++out_name=A01


cmd=python phc/run_hydra4.py learning=im_big2 exp_name=phc_prim_vr env=env_vr robot=smpl_humanoid env.motion_file=sample_data/amass_copycat_take5_train.pkl no_log=True env.num_envs=128 ++out_name=A01
module load singularity-wrapper && singularity_wrapper exec --mount type=bind,src="$(realpath .)",dst=/code --nv ./phc_latest.sif $cmd


