"""Serve a RoboCasa365 RLDX checkpoint using its native ZeroMQ protocol."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone-revision", default="4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7")
    args = parser.parse_args()

    os.environ.setdefault("RLDX_ATTN_IMPL", "sdpa")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    from rldx.data.embodiment_tags import EmbodimentTag
    from rldx.policy.rldx_policy import RLDXPolicy, RLDXSimPolicyWrapper
    from rldx.policy.server_client import PolicyServer

    policy = RLDXSimPolicyWrapper(RLDXPolicy(
        embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
        model_path=args.model_path, device=args.device,
        backbone_revision=args.backbone_revision,
    ))
    # The installed RC365 checkpoint omits image_max_area. Restore the
    # processor's documented geometry defaults before its first inference.
    processor = policy.policy.processor
    if processor.image_max_area is None or processor.image_resize_m is None:
        from rldx.data.augmentations import build_image_transformations_albumentations

        processor.image_max_area = processor.image_max_area or 65536
        processor.image_resize_m = processor.image_resize_m or 32
        processor.train_image_transform, processor.eval_image_transform = (
            build_image_transformations_albumentations(
                image_max_area=processor.image_max_area,
                image_resize_m=processor.image_resize_m,
                random_crop_fraction=getattr(processor, "random_crop_fraction", None),
                random_rotation_angle=getattr(processor, "random_rotation_angle", None),
                color_jitter_params=getattr(processor, "color_jitter_params", None),
            )
        )
    server = PolicyServer(policy, host=args.host, port=args.port)
    try:
        server.run()
    finally:
        server.socket.close(linger=0)
        server.context.term()


if __name__ == "__main__":
    main()
