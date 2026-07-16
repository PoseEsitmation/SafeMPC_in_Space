import torch


class Hparams():
    @staticmethod
    def add_hnet_hparams(hparams):
        # Hypernetwork
        if hparams.h_dims == [32, 32]:
            hparams.hnet_arch = [16, 16]
        elif hparams.h_dims == [200, 200]:
            hparams.hnet_arch = [50, 50]
        elif hparams.h_dims == [256, 256]:
            hparams.hnet_arch = [128, 128]
        elif hparams.h_dims == [200, 200, 200, 200]:
            hparams.hnet_arch = [256, 256]
        elif hparams.h_dims == [200, 200, 200]:
            hparams.hnet_arch = [100, 100]
        elif hparams.h_dims == [400, 400, 400]:
            hparams.hnet_arch = [100, 100]
        elif hparams.h_dims == [100, 100]:
            hparams.hnet_arch = [40, 40]

        if hparams.env == "door":
            hparams.hnet_act = "elu"
        elif hparams.env == "door_pose":
            hparams.hnet_act = "relu"
        elif hparams.env == "pusher":
            hparams.hnet_act = "elu"
        else:
            hparams.hnet_act = 'relu'

        # Embedding
        hparams.emb_size = 10
        # Initialization
        hparams.use_hyperfan_init = False
        hparams.hnet_init = "xavier"  # or "normal"
        hparams.std_normal_init = 0.02
        hparams.std_normal_temb = 1  # std when initializing task embedding

        # Training param
        hparams.lr_hyper = 0.0001
        hparams.grad_max_norm = 5

        if hparams.env == "door_pose" or hparams.env == "pusher_slide":
            hparams.beta = 0.5
        else:
            hparams.beta = 0.05

        hparams.no_look_ahead = False  # False=use two step optimization
        hparams.plastic_prev_tembs = False  # Allow adaptation of past task embeddings
        # Allow backpropagation through delta theta in the regularizer
        hparams.backprop_dt = False
        hparams.use_sgd_change = False  # Approximate change with in delta theta with SGD
        hparams.ewc_weight_importance = False  # Use fisher matrix to regularize
        # model weights generated from hnet
        hparams.n_fisher = -1  # Number of training samples to be used for the ' +
        # 'estimation of the diagonal Fisher elements. If ' +
        # "-1", all training samples are us

        hparams.si_eps = 1e-3
        hparams.mlp_var_minmax = True

        return hparams

    @staticmethod
    def add_chunked_hnet_hparams(hparams):
        # Hypernetwork
        if hparams.h_dims == [256, 256]:
            hparams.hnet_arch = [5, 5]
            hparams.chunk_dim = 12000  # Chunk size (output dim of hnet)
            hparams.cemb_size = 40
        elif hparams.h_dims == [200, 200, 200, 200]:
            hparams.hnet_arch = [25, 30]
            hparams.chunk_dim = 4000
            hparams.cemb_size = 20
        elif hparams.h_dims == [200, 200]:
            hparams.hnet_arch = [20, 20]
            hparams.chunk_dim = 2000
            hparams.cemb_size = 20
        hparams.hnet_act = 'relu'

        # Embedding
        hparams.emb_size = 10
        # Initialization
        hparams.use_hyperfan_init = False
        hparams.hnet_init = "xavier"  # or "normal"
        hparams.std_normal_init = 0.02
        hparams.std_normal_temb = 1  # std when initializing task embedding
        hparams.std_normal_cemb = 1

        # Training param
        hparams.lr_hyper = 0.0001
        hparams.grad_max_norm = 5
        hparams.beta = 0.005

        hparams.no_look_ahead = False  # False=use two step optimization
        hparams.plastic_prev_tembs = True  # Allow adaptation of past task embeddings
        # Allow backpropagation through delta theta in the regularizer
        hparams.backprop_dt = False
        hparams.use_sgd_change = False  # Approximate change with in delta theta with SGD
        hparams.ewc_weight_importance = False  # Use fisher matrix to regularize
        # model weights generated from hnet
        hparams.n_fisher = -1  # Number of training samples to be used for the ' +
        # 'estimation of the diagonal Fisher elements. If ' +
        # "-1", all training samples are us

        return hparams


def default_arg_policy(hparams):
    """Policy network (imitation learning) hyperparameters, shared across envs."""
    hparams.policy_lr = 1e-4
    hparams.policy_bs = 128
    hparams.policy_train_iters = 1000   # training iterations per dynamics update
    hparams.policy_lambda_imit = 1.0    # weight on imitation (BC) loss
    hparams.policy_lambda_cbf = 2.0    # weight on CBF penalty loss (Eq. 16)
    hparams.policy_lambda_clf = 2.0    # weight on CLF penalty loss (Eq. 17)
    # multiplier applied to lambda_cbf/clf each DAGGER iter
    hparams.policy_lambda_ramp = 2.0
    hparams.policy_lambda_max = 1.0    # curriculum cap for lambda_cbf/clf
    # set True in env configs that have CBF/CLF
    hparams.policy_use_safety_loss = False
    hparams.policy_train_start = 0      # MPC steps before policy training begins
    hparams.dagger_every = 0      # run DAGGER every N MPC steps (0 = disabled)
    hparams.dagger_n_iter = 5      # number of DAGGER refinement iterations total
    hparams.dagger_n_rollout = 5      # rollout episodes per DAGGER iteration
    # Execute raw mixed actions during DAGGER rollouts (False, the default).
    # True filters them through the safety filter, which keeps rollouts away
    # from exactly the states where the raw policy fails, so the buffer never
    # contains avoidance labels — see PolicyTrainer.dagger_update.
    hparams.dagger_filter_rollouts = False
    # Post-DAGGER validation episodes (per DAGGER iteration, hnet_exp).  The
    # per-episode metrics are high-variance (init attitude error 80–180°,
    # random KOZ placement/size per reset), so 3 episodes made the
    # dagger_eval_* curves jump around.  Unfiltered episodes are nearly free
    # (~0.1 s: raw env + policy forward) and carry the key safety signal
    # (KOZ-violation decline) → many.  Filtered episodes solve the QP every
    # step (~6 s/episode measured) and carry the filter-activation-decline
    # signal → fewer.  The same counts are used for the pre-training
    # baseline eval (round 0, untrained policy).
    hparams.dagger_val_eps_unfiltered = 40
    # 15 filtered episodes (was 10): the intervention rate is a rare-event
    # estimate and its round-to-round scatter was the main noise source in
    # the reliance-decline chart (paper_final2).  ~+30 s per round.
    hparams.dagger_val_eps_filtered = 15
    # Safety-prioritised sampling for policy training: rows where the expert's
    # filter corrected the label, and rows within policy_safety_margin_deg of
    # the KOZ boundary, are drawn up to policy_safety_oversample x more often
    # (WeightedRandomSampler).  1.0 = uniform sampling (off).
    hparams.policy_safety_oversample = 1.0
    hparams.policy_safety_margin_deg = 15.0
    # CBF hinge margin during policy training (0 = penalise only outright
    # violation).  Positive values create gradient in the approach corridor
    # and give the learned condition robustness headroom.
    hparams.policy_cbf_eps_train = 0.0
    # Fraction of DAGGER rollout episodes run with the pure NN policy (κ=0)
    # so the buffer contains the learner's own failure states (0 = all
    # episodes follow the κ curriculum).
    hparams.dagger_student_frac = 0.0
    return hparams


def default_arg_half_cheetah_safe(hparams):
    hparams.state_dim = 19  # 18 base + x_pos appended at obs[18]
    hparams.control_dim = 6
    hparams.out_dim = hparams.state_dim
    hparams.policy_lambda_cbf = 1e-4   # activate CBF loss (x_pos now in obs)
    hparams.policy_use_safety_loss = True

    # Tasks
    hparams.num_tasks = 3
    hparams.max_iteration = 100000          # was 100000 3000
    hparams.init_rand_steps = 10000         # was 10000  400
    hparams.dynamics_update_every = 1000   # was 1000   200
    # skip first policy round (random-data only)
    hparams.policy_train_start = 1000

    # Dynamics model
    hparams.dnn_out = "diff"
    hparams.normalize_xu = True
    # was [200,200,200,200] — smaller = faster
    hparams.h_dims = [256, 256]
    hparams.out_var = True               # was True — disables expensive variance head

    hparams.lr = 0.001
    hparams.lr_steps = None
    hparams.bs = 100                       # was 100
    hparams.reg_lambda = 0.00005
    hparams.train_dynamic_iters = 2000     # was 2000
    hparams.print_train_every = 500
    hparams.eval_every = 500              # was 2000

    hparams.eval_env_run_every = 5000      # was 5000
    hparams.run_eval_env_eps = 1
    hparams.M = 30                        # was 1000

    # MPC-CEM planner
    hparams.control = "mpc-cem"
    hparams.horizon = 25                  # was 30
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM — these are the biggest CPU cost
    hparams.n_sim_steps = 5
    hparams.n_sim_particles = 400         # was 500
    hparams.num_cem_elites = 40           # was 50

    hparams.mag_noise = 1

    return hparams


def HP(env, seed=None, save_folder='./runs/lqr', run_name=None):
    hparams = Hparams()
    hparams.seed = seed if seed is not None else 2020
    hparams.save_folder = save_folder if save_folder is not None else './runs/lqr'
    hparams.run_name = run_name or ""
    hparams.resume = False

    # Common train setting
    hparams.num_ds_worker = 0
    hparams.print_train_every = 1000
    hparams.save_every = 1000

    # common RL setting
    hparams.env = env
    hparams.gt_dynamic = False
    hparams.device = (
        "cuda:0" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Policy (imitation learning) hyperparams — applied to every env
    hparams = default_arg_policy(hparams)

    if env == "lqr":
        return default_arg_2d_car(hparams)
    elif env == "lqr10":
        return default_arg_10d_car(hparams)
    elif env.startswith("hopper"):
        return default_arg_hopper(hparams)
    elif env == "humanoid":
        return default_arg_humanoid(hparams)
    elif env.startswith("half_cheetah_safe"):
        return default_arg_half_cheetah_safe(hparams)
    elif env.startswith("half_cheetah"):
        return default_arg_half_cheetah(hparams)
    elif env.startswith("inverted_pendulum"):
        return default_arg_inverted_pendulum(hparams)
    elif env.startswith("pendulum"):
        return default_arg_pendulum(hparams)
    elif env == "cartpole":
        return default_arg_cartpole(hparams)
    elif env == "cartpole_bin":
        return default_arg_cartpole_bin(hparams)
    elif env == "metaworld10":
        return default_arg_metaworld10(hparams)
    elif env == "reacher":
        return default_arg_reacher(hparams)
    elif env == "pusher":
        return default_arg_pusher(hparams)
    elif env == "door":
        return default_arg_door(hparams)
    elif env == "door_pose":
        return default_arg_door_pose(hparams)
    elif env == "pusher_rot":
        return default_arg_pusher_rot(hparams)
    elif env == "pusher_slide":
        return default_arg_pusher_slide(hparams)
    elif env.startswith("spaceEnv"):
        return default_arg_sat(hparams)


def default_arg_metaworld10(hparams):
    hparams.state_dim = 9
    hparams.control_dim = 4
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 3
    hparams.max_iteration = 30000
    hparams.init_rand_steps = 10000
    hparams.dynamics_update_every = 1500

    # Common Dynamics Model
    hparams.dnn_out = "diff"  # or "diff"
    hparams.normalize_xu = True
    hparams.h_dims = [256, 256]
    hparams.out_var = False

    hparams.lr = 0.001
    hparams.lr_steps = None
    hparams.bs = 100
    hparams.reg_lambda = 0.0001
    hparams.train_dynamic_iters = 10000
    hparams.eval_every = 5000

    # RL Eval setting
    hparams.eval_env_run_every = 1500
    hparams.run_eval_env_eps = 5

    # Size of inducing points
    hparams.M = 400

    # RL Planning
    hparams.control = "mpc-pddm"
    hparams.horizon = 7
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5  # Number of search steps
    hparams.n_sim_particles = 2000  # Number of traj to sample
    hparams.num_cem_elites = 10

    # PDDM
    hparams.pddm_beta = 0.8
    hparams.pddm_kappa = 20
    hparams.mag_noise = 1

    return hparams


def default_arg_humanoid(hparams):
    hparams.state_dim = 376
    hparams.control_dim = 17
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 5
    hparams.max_iteration = 40001
    hparams.init_rand_steps = 1000
    hparams.dynamics_update_every = 10000

    # Common Dynamics Model
    hparams.dnn_out = "state"  # or "diff"
    hparams.normalize_xu = True
    hparams.h_dims = [256, 256]
    hparams.out_var = False

    hparams.lr = 0.0001
    hparams.lr_steps = [8500]
    hparams.bs = 100
    hparams.reg_lambda = 0
    hparams.train_dynamic_iters = 10000
    hparams.eval_every = 5000

    # Size of inducing points
    hparams.M = 50

    # RL Eval setting
    hparams.eval_env_run_every = 200
    hparams.run_eval_env_eps = 5

    # RL Planning
    hparams.control = "mpc-mppi"
    hparams.horizon = 20
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99
    hparams.mag_noise = 1
    hparams.pddm_kappa = 20
    hparams.pddm_beta = 0.5

    # CEM
    hparams.n_sim_steps = 10  # Number of search steps
    hparams.n_sim_particles = 100  # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 5

    return hparams


def default_arg_hopper(hparams):
    hparams.state_dim = 12
    hparams.control_dim = 3
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 3
    hparams.max_iteration = 100000
    hparams.init_rand_steps = 10000
    hparams.dynamics_update_every = 1000

    # Common Dynamics Model
    hparams.dnn_out = "diff"  # or "diff"
    hparams.normalize_xu = True
    hparams.h_dims = [200, 200, 200, 200]
    hparams.out_var = True

    hparams.lr = 0.001
    hparams.lr_steps = None
    hparams.bs = 100
    hparams.reg_lambda = 0.000075
    hparams.train_dynamic_iters = 2000
    hparams.eval_every = 2000

    # RL Eval setting
    hparams.eval_env_run_every = 5000
    hparams.run_eval_env_eps = 4

    # Size of inducing points
    hparams.M = 1000

    # RL Planning
    hparams.control = "mpc-pddm"
    hparams.horizon = 7
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5  # Number of search steps
    hparams.n_sim_particles = 2500  # Number of traj to sample
    hparams.num_cem_elites = 50

    # PDDM
    hparams.pddm_beta = 0.7
    hparams.pddm_kappa = 20
    hparams.mag_noise = 1

    return hparams


def default_arg_pendulum(hparams):
    hparams.state_dim = 3
    hparams.control_dim = 1
    hparams.out_dim = hparams.state_dim
    # Tasks
    hparams.num_tasks = 5
    hparams.init_rand_steps = 400
    hparams.max_iteration = 10000
    hparams.dynamics_update_every = 400

    # Common Dynamics Model
    hparams.dnn_out = "diff"  # or "state"
    hparams.normalize_xu = False
    hparams.h_dims = [32, 32]
    hparams.out_var = False

    hparams.lr = 0.001
    hparams.lr_steps = None  # learning rate decay steps
    hparams.bs = 20
    hparams.reg_lambda = 0
    hparams.train_dynamic_iters = 1000

    # Central Device
    if not hasattr(hparams, "device"):
        hparams.device = "cpu"
    hparams.eval_every = 500

    # Size of inducing points
    hparams.M = 50

    # RL Eval setting
    hparams.eval_env_run_every = 200
    hparams.run_eval_env_eps = 5

    # RL Planning
    hparams.control = "mpc-mppi"
    hparams.horizon = 15
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 10  # Number of search steps
    hparams.n_sim_particles = 100  # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 5

    return hparams


def default_arg_inverted_pendulum(hparams):
    hparams.state_dim = 4
    hparams.control_dim = 1
    hparams.out_dim = hparams.state_dim
    # Tasks
    hparams.num_tasks = 3
    hparams.init_rand_steps = 2000
    hparams.max_iteration = 40000
    hparams.dynamics_update_every = 1000
    hparams.out_var = False

    # Common Dynamics Model
    hparams.dnn_out = "diff"  # or "state"
    hparams.normalize_xu = False
    hparams.h_dims = [256, 256]

    hparams.lr = 0.001
    hparams.lr_steps = None  # learning rate decay steps
    hparams.bs = 100
    hparams.reg_lambda = 0.0001
    hparams.train_dynamic_iters = 2000
    # Central Device
    if not hasattr(hparams, "device"):
        hparams.device = "cpu"
    hparams.eval_every = 2000

    # Size of inducing points
    hparams.M = 400

    # RL Eval setting
    hparams.eval_env_run_every = 4000
    hparams.run_eval_env_eps = 4

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 25
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5  # Number of search steps
    hparams.n_sim_particles = 1000  # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 10

    return hparams


def default_arg_half_cheetah(hparams):
    hparams.state_dim = 18
    hparams.control_dim = 6
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 3
    hparams.max_iteration = 100000
    hparams.init_rand_steps = 10000
    hparams.dynamics_update_every = 1000

    # Common Dynamics Model
    hparams.dnn_out = "diff"  # or "diff"
    hparams.normalize_xu = True
    hparams.h_dims = [200, 200, 200, 200]
    hparams.out_var = True

    hparams.lr = 0.001
    hparams.lr_steps = None
    hparams.bs = 100
    hparams.reg_lambda = 0.000075
    hparams.train_dynamic_iters = 2000
    hparams.eval_every = 2000

    # RL Eval setting
    hparams.eval_env_run_every = 5000
    hparams.run_eval_env_eps = 1

    # Size of inducing points
    hparams.M = 1000

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 30
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5  # Number of search steps
    hparams.n_sim_particles = 500  # Number of traj to sample
    hparams.num_cem_elites = 50

    # PDDM
    hparams.pddm_beta = 0.7
    hparams.pddm_kappa = 20
    hparams.mag_noise = 1

    return hparams


def default_arg_cartpole(hparams):
    hparams.state_dim = 4
    hparams.control_dim = 1
    hparams.out_dim = hparams.state_dim
    # Tasks
    hparams.num_tasks = 10
    hparams.init_rand_steps = 400
    hparams.max_iteration = 3000
    hparams.dynamics_update_every = 200
    hparams.out_var = False

    # Common Dynamics Model
    hparams.dnn_out = "diff"  # or "state"
    hparams.normalize_xu = True
    hparams.h_dims = [256, 256]

    hparams.lr = 0.001
    hparams.lr_steps = None  # learning rate decay steps
    hparams.bs = 32
    hparams.reg_lambda = 0.00005
    hparams.train_dynamic_iters = 500
    hparams.print_train_every = 500
    # Central Device
    if not hasattr(hparams, "device"):
        hparams.device = "cpu"
    hparams.eval_every = 500

    # Size of inducing points
    hparams.M = 30

    # RL Eval setting
    hparams.eval_env_run_every = 200
    hparams.run_eval_env_eps = 1

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 25
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5  # Number of search steps
    hparams.n_sim_particles = 400  # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 40

    hparams.mag_noise = 1

    return hparams


def default_arg_cartpole_bin(hparams):
    hparams.state_dim = 4
    hparams.control_dim = 1
    hparams.out_dim = hparams.state_dim
    # Tasks
    hparams.num_tasks = 3
    hparams.init_rand_steps = 2000
    hparams.max_iteration = 5000
    hparams.dynamics_update_every = 200
    hparams.out_var = False

    # Common Dynamics Model
    hparams.dnn_out = "diff"  # or "state"
    hparams.normalize_xu = False
    hparams.h_dims = [256, 256]

    hparams.lr = 0.001
    hparams.lr_steps = None  # learning rate decay steps
    hparams.bs = 32
    hparams.reg_lambda = 0.00005
    hparams.train_dynamic_iters = 500
    hparams.print_train_every = 500
    # Central Device
    if not hasattr(hparams, "device"):
        hparams.device = "cpu"
    hparams.eval_every = 500

    # Size of inducing points
    hparams.M = 200

    # RL Eval setting
    hparams.eval_env_run_every = 1000
    hparams.run_eval_env_eps = 1

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 25
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5  # Number of search steps
    hparams.n_sim_particles = 400  # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 40

    return hparams


def default_arg_2d_car(hparams):
    hparams.state_dim = 4
    hparams.control_dim = 2
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 2
    hparams.max_iteration = 4000
    hparams.init_rand_steps = 2000
    hparams.dynamics_update_every = 200

    # Common Dynamics Model
    hparams.dnn_out = "state"  # or "diff"
    hparams.normalize_xu = True
    hparams.h_dims = [32, 32]
    hparams.out_var = False

    hparams.lr = 0.001
    hparams.lr_steps = [4500]
    hparams.bs = 100
    hparams.reg_lambda = 0
    hparams.train_dynamic_iters = 5000
    hparams.eval_every = 2500

    # Size of inducing points
    hparams.M = 50

    # RL Planning
    hparams.control = "mpc-mppi"
    hparams.horizon = 200
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # RL Eval setting
    hparams.eval_env_run_every = 200
    hparams.run_eval_env_eps = 5

    # CEM
    hparams.n_sim_steps = 10  # Number of search steps
    hparams.n_sim_particles = 100  # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 5

    return hparams


def default_arg_10d_car(hparams):
    hparams.state_dim = 20
    hparams.control_dim = 10
    hparams.out_dim = hparams.state_dim
    hparams.rand_aggregate_seed = 2020

    # Tasks
    hparams.num_tasks = 4
    hparams.max_iteration = 1
    hparams.init_rand_steps = 10000
    hparams.dynamics_update_every = 400
    hparams.out_var = False

    # Common Dynamics Model
    hparams.dnn_out = "diff"  # or "diff"
    hparams.normalize_xu = False
    hparams.h_dims = [32, 32]

    hparams.lr = 0.0001
    hparams.lr_steps = None
    hparams.bs = 100
    hparams.reg_lambda = 0
    hparams.train_dynamic_iters = 50000
    hparams.eval_every = 2500

    # Size of inducing points
    hparams.M = 50

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 30
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # RL Eval setting
    hparams.eval_env_run_every = 400
    hparams.run_eval_env_eps = 1

    # CEM
    hparams.n_sim_steps = 5  # Number of search steps
    # Number of traj to sample(in cem and mppi)
    hparams.n_sim_particles = 10000
    hparams.num_cem_elites = 5

    return hparams


def default_arg_reacher(hparams):
    hparams.state_dim = 11
    hparams.control_dim = 2
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 10
    hparams.init_rand_steps = 200
    hparams.max_iteration = 3000
    hparams.dynamics_update_every = 50
    hparams.out_var = False

    # Common Dynamics Model
    hparams.dnn_out = "diff"  # or "state"
    hparams.normalize_xu = True
    hparams.h_dims = [256, 256]

    hparams.lr = 0.001
    hparams.lr_steps = None  # learning rate decay steps
    hparams.bs = 32
    hparams.reg_lambda = 0.00005
    hparams.train_dynamic_iters = 150
    hparams.print_train_every = 150
    # Central Device
    if not hasattr(hparams, "device"):
        hparams.device = "cpu"
    hparams.eval_every = 150

    # Size of inducing points
    hparams.M = 30

    # RL Eval setting
    hparams.eval_env_run_every = 200
    hparams.run_eval_env_eps = 4

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 25
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5  # Number of search steps
    hparams.n_sim_particles = 400  # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 40

    # PDDM
    hparams.pddm_beta = 0.6
    hparams.pddm_kappa = 10
    hparams.mag_noise = 1

    return hparams


def default_arg_pusher(hparams):
    hparams.state_dim = 10
    hparams.control_dim = 2
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 5
    hparams.init_rand_steps = 2000
    hparams.max_iteration = 4000
    hparams.dynamics_update_every = 200
    hparams.out_var = False

    # Common Dynamics Model
    hparams.dnn_out = "diff"  # or "state"
    hparams.normalize_xu = True
    hparams.h_dims = [200, 200]

    hparams.lr = 0.001
    hparams.lr_steps = None  # learning rate decay steps
    hparams.bs = 100
    hparams.reg_lambda = 0
    hparams.train_dynamic_iters = 2000
    hparams.print_train_every = 500
    # Central Device
    if not hasattr(hparams, "device"):
        hparams.device = "cpu"
    hparams.eval_every = 1000

    # Size of inducing points
    hparams.M = 100

    # RL Eval setting
    hparams.eval_env_run_every = 200
    hparams.run_eval_env_eps = 5

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 20
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5  # Number of search steps
    hparams.n_sim_particles = 500  # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 40

    # PDDM
    hparams.pddm_beta = 0.6
    hparams.pddm_kappa = 50
    hparams.mag_noise = 1.0

    return hparams


def default_arg_pusher_rot(hparams):
    hparams.state_dim = 20
    hparams.control_dim = 2
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 5
    hparams.init_rand_steps = 2000
    hparams.max_iteration = 4000
    hparams.dynamics_update_every = 200
    hparams.out_var = False

    # Common Dynamics Model
    hparams.dnn_out = "diff"  # or "state"
    hparams.normalize_xu = True
    hparams.h_dims = [200, 200]

    hparams.lr = 0.001
    hparams.lr_steps = None  # learning rate decay steps
    hparams.bs = 100
    hparams.reg_lambda = 0
    hparams.train_dynamic_iters = 2000
    hparams.print_train_every = 500
    # Central Device
    if not hasattr(hparams, "device"):
        hparams.device = "cpu"
    hparams.eval_every = 1000

    # Size of inducing points
    hparams.M = 100

    # RL Eval setting
    hparams.eval_env_run_every = 200
    hparams.run_eval_env_eps = 5

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 20
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5  # Number of search steps
    hparams.n_sim_particles = 500  # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 40

    # PDDM
    hparams.pddm_beta = 0.6
    hparams.pddm_kappa = 50
    hparams.mag_noise = 1.0

    return hparams


def default_arg_pusher_slide(hparams):
    hparams.state_dim = 18
    hparams.control_dim = 2
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 5
    hparams.init_rand_steps = 300
    hparams.max_iteration = 3000
    hparams.dynamics_update_every = 150
    hparams.out_var = False

    # Common Dynamics Model
    hparams.dnn_out = "diff"  # or "state"
    hparams.normalize_xu = True
    hparams.h_dims = [200, 200]

    hparams.lr = 0.001
    hparams.lr_steps = None  # learning rate decay steps
    hparams.bs = 100
    hparams.reg_lambda = 0
    hparams.train_dynamic_iters = 500
    hparams.print_train_every = 500
    # Central Device
    if not hasattr(hparams, "device"):
        hparams.device = "cpu"
    hparams.eval_every = 1000

    # Size of inducing points
    hparams.M = 100

    # RL Eval setting
    hparams.eval_env_run_every = 200
    hparams.run_eval_env_eps = 5

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 20
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5  # Number of search steps
    hparams.n_sim_particles = 500  # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 40

    # PDDM
    hparams.pddm_beta = 0.6
    hparams.pddm_kappa = 50
    hparams.mag_noise = 1.0

    return hparams


def default_arg_door(hparams):
    hparams.state_dim = 4
    hparams.control_dim = 3
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 1
    hparams.init_rand_steps = 2000
    hparams.max_iteration = 4000
    hparams.dynamics_update_every = 200
    hparams.out_var = False

    # Common Dynamics Model
    hparams.dnn_out = "diff"  # or "state"
    hparams.normalize_xu = True
    hparams.h_dims = [200, 200]

    hparams.lr = 0.001
    hparams.lr_steps = None  # learning rate decay steps
    hparams.bs = 100
    hparams.reg_lambda = 0
    hparams.train_dynamic_iters = 2000
    hparams.print_train_every = 500
    # Central Device
    if not hasattr(hparams, "device"):
        hparams.device = "cpu"
    hparams.eval_every = 1000

    # Size of inducing points
    hparams.M = 100

    # RL Eval setting
    hparams.eval_env_run_every = 1000
    hparams.run_eval_env_eps = 4

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 20
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5  # Number of search steps
    hparams.n_sim_particles = 500  # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 40

    # PDDM
    hparams.pddm_beta = 0.6
    hparams.pddm_kappa = 50
    hparams.mag_noise = 1.0

    return hparams


def default_arg_door_pose(hparams):
    hparams.state_dim = 26
    hparams.control_dim = 7
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 5
    hparams.init_rand_steps = 2000
    hparams.max_iteration = 60000
    hparams.dynamics_update_every = 200
    hparams.out_var = True

    # Common Dynamics Model
    hparams.dnn_out = "diff"  # or "state"
    hparams.normalize_xu = True
    hparams.h_dims = [200, 200, 200, 200]

    hparams.lr = 0.001
    hparams.lr_steps = None  # learning rate decay steps
    hparams.bs = 100
    hparams.reg_lambda = 0.00001
    hparams.train_dynamic_iters = 200
    hparams.print_train_every = 200
    # Central Device
    if not hasattr(hparams, "device"):
        hparams.device = "cpu"
    hparams.eval_every = 200

    # Size of inducing points
    hparams.M = 600

    # RL Eval setting
    hparams.eval_env_run_every = 1000
    hparams.run_eval_env_eps = 1

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 10
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5  # Number of search steps
    hparams.n_sim_particles = 2000  # Number of traj to sample(in cem and mppi)
    hparams.num_cem_elites = 40

    # PDDM
    hparams.pddm_beta = 0.6
    hparams.pddm_kappa = 50
    hparams.mag_noise = 0.5

    return hparams


def default_arg_sat(hparams):
    hparams.state_dim = 13
    hparams.control_dim = 3
    hparams.out_dim = hparams.state_dim

    # Tasks
    hparams.num_tasks = 4
    # 5000 random steps before MPC — matches the confirmed-good pure-MBRL run
    # (20260618_115015) and stabilises the x/u/diff normalisation statistics
    # before anything downstream consumes them.
    hparams.init_rand_steps = 5000
    # 20000 (was 30000 in baseline_20): shortened for faster validation
    # iterations.  The post-policy_train_start phase shrinks from 20000 to
    # 10000 steps; dagger_every drops to 1000 below so all 10 DAGGER
    # curriculum sessions still complete before the run ends.
    hparams.max_iteration = 20000
    hparams.dynamics_update_every = 500
    # The MPC expert needs several episodes of model learning before its
    # labels are worth imitating (baseline_16/17: starting at 2000 trained on
    # garbage).  Under the FIXED SCENARIO the expert is goal-reaching from
    # episode 3-4 (paper_final: att_err 2.1° at ep 3, reward 1237 at ep 4), so
    # the old 10000 gate wasted half the run: 6000 starts BC/DAGGER after ~6
    # competent-ish episodes and yields 28 DAGGER rounds instead of 20.
    hparams.policy_train_start = 6000

    # Common Dynamics Model
    hparams.dnn_out = "diff"
    hparams.normalize_xu = True
    # Satellite dynamics are stiff: per-step omega/quaternion changes are
    # orders of magnitude smaller than theta changes. Without normalizing the
    # diff target the model ignores omega entirely (R^2 < 0) and the planner
    # has no usable action->state signal. See datautil.normalize_diff.
    hparams.normalize_diff = True
    # Freeze x/diff normalisation stats at the end of task 0's random phase
    # (actions use the identity transform — the box is already [-1,1]³).  One
    # fixed coordinate system for the whole run: normalised DAGGER buffers and
    # CBF/CLF closures can't go stale, and the hnet trains on a stationary
    # input distribution shared across tasks (task differences must then be
    # expressed through the task embedding, not through per-task norms).
    # Stats are saved as norms.pt in the run dir; reuse them in another run
    # with --norms-path to get identical coordinates across runs.
    hparams.freeze_norms = True
    hparams.h_dims = [256, 256]
    hparams.out_var = False

    hparams.lr = 0.001
    hparams.lr_steps = None
    hparams.bs = 100
    hparams.reg_lambda = 0.0001
    hparams.train_dynamic_iters = 2000
    hparams.eval_every = 2000

    # Size of inducing points
    hparams.M = 400

    # RL Eval setting
    hparams.eval_env_run_every = 5000
    hparams.run_eval_env_eps = 3

    # RL Planning
    hparams.control = "mpc-cem"
    hparams.horizon = 15
    hparams.propagation = "EP"
    hparams.reward_discount = 0.99

    # CEM
    hparams.n_sim_steps = 5
    hparams.n_sim_particles = 500
    hparams.num_cem_elites = 50
    # Clip CEM samples to the physical actuator box [-1, 1]³ so the planner
    # only considers executable torques (see MPC.__init__).
    hparams.mpc_u_bound = 1.0

    # PDDM
    hparams.pddm_beta = 0.7
    hparams.pddm_kappa = 20
    hparams.mag_noise = 1.0
    hparams.policy_use_safety_loss = True
    # CBF/CLF curriculum (paper Sec. IV-A: weights "designed to yield losses
    # of similar order of magnitude").  Measured on safe expert states:
    # loss_imit ≈ 2-3, loss_cbf ≈ 1e-3, loss_clf ≈ 5e-4 — matching imitation
    # at full curriculum strength therefore needs λ ≈ O(10³).  baseline_21
    # (cap 100 → peak CBF contribution ≈ 0.1) left cbf_viol_frac pinned at
    # 20-40% and unfiltered validation KOZ violations flat across all 10
    # DAGGER iterations.  Cap 500 puts safe-state batches just below
    # imitation while letting batches that contain actual near-KOZ states
    # (which unfiltered DAGGER rollouts now supply, with per-sample hinges
    # orders larger) dominate the gradient — by design: safety first.
    # Start λ_cbf at 8 (was 1.0): paper_run_2 showed the mid-curriculum hump —
    # BC learns the expert's aggressive slews (speed) before precision, and
    # with λ_cbf at 1-8 for the first 3-4 DAGGER iters the safety term was
    # negligible against loss_imit ≈ 0.13, so filtered-eval violations climbed
    # (0 → 5.9 → 49.4/ep) while the policy out-ran the one-step QP into the
    # hard-infeasible regime (fallback bursts of 300-440 steps).  Starting at
    # 8 puts CBF pressure on the very first BC round.
    hparams.policy_lambda_cbf = 8.0
    hparams.policy_lambda_clf = 0.05
    # ×2 per DAGGER iter (paper Algorithm 1 line 19).  The old ×3 hit the cap
    # by iter ~6, so ~70% of the curriculum ran with the boundary-CBF term at
    # full strength against imitation — the suspected driver of baseline_34's
    # mid-run regression (koz 1.3 at iter 8 → 37.8 at iter 11).  With start 8
    # and doubling, the 500 cap arrives at iter ~7 of 10 (fast profile).
    hparams.policy_lambda_ramp = 2.0
    hparams.policy_lambda_max = 500.0
    hparams.dagger_every = 500    # DAGGER at steps 6000, 6500, ..., 19500
    hparams.dagger_n_iter = 28     # (20000-6000)/500 — κ anneals 0.96 → 0.0
    # 3 rollouts/round (was 5): with 40% more rounds this keeps total expert-
    # labelling cost flat while giving a finer κ/λ curriculum and 8 more
    # retrain+validation points; student episodes: round(0.4*3) = 1 per round.
    hparams.dagger_n_rollout = 3
    # Oversample safety-critical rows ~10x: expert-filter-corrected labels
    # (buffer tags) and states within 15° of the KOZ (θ-margin, all rows) —
    # they are ~a few % of the data but carry all the avoidance knowledge.
    hparams.policy_safety_oversample = 10.0
    # baseline_33 (loss_cbf == 0.0000 all run, unfiltered KOZ violations flat
    # across 20 DAGGER iters): the CBF penalty needs states it can fire on
    # and a margin to fire early.  ε_train = 0.05 ≈ p(≈30%) of the observed
    # condition-margin distribution (mean 0.17); the boundary sampler (set in
    # hnet_exp) supplies synthetic corridor states; student rollouts harvest
    # the policy's real failure states with expert labels.
    hparams.policy_cbf_eps_train = 0.05
    hparams.dagger_student_frac = 0.4   # 2 of 5 rollout episodes pure NN

    return hparams
