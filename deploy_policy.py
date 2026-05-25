# import packages and module here
import sys, os
from .model import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def encode_obs(observation):  # Post-Process Observation
    observation["agent_pos"] = observation["joint_action"]["vector"]
    return observation


def _select_high_view_rgb(observation, preferred_source="third_view_rgb"):
    source_order = {
        "third_view_rgb": ("third_view_rgb", "head_camera"),
        "head_camera": ("head_camera", "third_view_rgb"),
        "auto": ("third_view_rgb", "head_camera"),
    }.get(preferred_source, (preferred_source, "third_view_rgb", "head_camera"))

    for source in source_order:
        if source == "third_view_rgb":
            third_view_rgb = observation.get("third_view_rgb")
            if third_view_rgb is not None:
                return third_view_rgb, source
        elif source == "head_camera":
            head_camera = observation.get("observation", {}).get("head_camera", {})
            if "rgb" in head_camera:
                return head_camera["rgb"], source

    raise KeyError(
        f"Could not find a valid high-view image in observation for preferred source '{preferred_source}'."
    )


def _collect_policy_inputs(observation, preferred_high_view_source="third_view_rgb"):
    high_view_rgb, high_view_source = _select_high_view_rgb(observation, preferred_high_view_source)
    wrist_obs = observation["observation"]
    input_rgb_arr = [
        high_view_rgb,
        wrist_obs["right_camera"]["rgb"],
        wrist_obs["left_camera"]["rgb"],
    ]
    return input_rgb_arr, observation["agent_pos"], high_view_source


def get_model(usr_args):  # keep
    model_name = usr_args["ckpt_setting"]
    checkpoint_id = usr_args["checkpoint_id"]
    left_arm_dim, right_arm_dim, rdt_step = (
        usr_args["left_arm_dim"],
        usr_args["right_arm_dim"],
        usr_args["rdt_step"],
    )
    
    checkpoint_root = os.environ.get(
        "FRAPPE_CHECKPOINT_ROOT",
        os.path.join(parent_directory, "checkpoints"),
    )
    use_ema = usr_args.get("use_ema", False)
    if use_ema:
        main_checkpoint_path = os.path.join(
            checkpoint_root,
            model_name,
            f"checkpoint-{checkpoint_id}",
        )
        ema_checkpoint_path = os.path.join(
            checkpoint_root,
            model_name,
            f"checkpoint-{checkpoint_id}/ema",
        )
        print(f"Loading EMA model from: {ema_checkpoint_path}")
        print(f"Using config from: {main_checkpoint_path}")
        checkpoint_path = main_checkpoint_path
        usr_args["ema_model_path"] = ema_checkpoint_path
    else:
        checkpoint_path = os.path.join(
            checkpoint_root,
            model_name,
            f"checkpoint-{checkpoint_id}",
        )
        print(f"Loading main model from: {checkpoint_path}")
    
    rdt = RDT(
        checkpoint_path,
        usr_args["task_name"],
        left_arm_dim,
        right_arm_dim,
        rdt_step,
        ema_model_path=usr_args.get("ema_model_path"),
        encoder_weights_root=usr_args.get("encoder_weights_root"),
    )
    rdt.preferred_high_view_source = usr_args.get("high_view_source", "third_view_rgb")
    
    return rdt


def eval(TASK_ENV, model, observation):
    """x
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    obs = encode_obs(observation)  # Post-Process Observation
    instruction = TASK_ENV.get_instruction()
    input_rgb_arr, input_state, high_view_source = _collect_policy_inputs(
        obs,
        getattr(model, "preferred_high_view_source", "third_view_rgb"),
    )

    if (model.observation_window
            is None):  # Force an update of the observation at the first frame to avoid an empty observation window
        print(f"Using {high_view_source} as cam_high input for evaluation.")
        model.set_language_instruction(instruction)
        model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action()  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        input_rgb_arr, input_state, _ = _collect_policy_inputs(
            obs,
            getattr(model, "preferred_high_view_source", "third_view_rgb"),
        )
        model.update_observation_window(input_rgb_arr, input_state)  # Update Observation


def reset_model(
        model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.reset_obsrvationwindows()
