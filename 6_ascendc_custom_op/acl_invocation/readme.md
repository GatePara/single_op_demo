# acl samples
```shell
cd acl_offline_model/acl_online_model
bash run_torch.sh ${is_dynamic}(0/1) ${replay_mode}(/batch/iterator)
```

# run static op (depend on chip version)
```shell
(cd acl_offline_model; bash run.sh --is-dynamic 0)

(cd acl_online_model; bash run.sh --is-dynamic 0)

(cd acl_online_model_unalign; bash run.sh --is-dynamic 0)
```

# run dynamic op (depend on chip version)
```shell
(cd acl_offline_model; bash run.sh --is-dynamic 1)

(cd acl_online_model; bash run.sh --is-dynamic 1)
```
