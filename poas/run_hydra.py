import hydra

import toy_grid_dag


@hydra.main(config_path="configs", config_name="main") # use hydra==1.1
def main(cfg):
    class ARGS:
        pass
    args = ARGS()
    for k, v in cfg.items():
        setattr(args, k, v)
    args.log_name = f"dim_{args.ndim}_r0{args.R0}_horizon{args.horizon}_seed{args.seed}_{args.method}"
    print(args.log_name)
    # set hydra output dir to the same as the log dir
    # print hydra current work dir
    print(hydra.utils.get_original_cwd())
    # change output dir for hydra 

    toy_grid_dag.main(args)

if __name__ == "__main__":
    main()