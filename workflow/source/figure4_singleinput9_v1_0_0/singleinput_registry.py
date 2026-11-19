#!/usr/bin/env python3
"""Register the already completed, fixed nine GatedConv single-input checkpoints."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
from figure4_common import CLASSES, HERE, VENDOR, identity_hash, read_json, sha256, verify_run

SEEDS = (42, 2024, 2025)
SCHEMES = ('raw_only', 'context_only', 'responses_only')
MODEL = 'gatedconv_target'
RUNTIME_SHA256 = {
    'model_benchmark_all.py': 'ca14b49d514a95dfa3443f019bb60d475ed5cf2feb7e1cd161d42ea000b8f806',
    'IterDataset.py': 'fe34d1330600610dbebb11de1a30752123a2108ce4d435621cd8f7cf128e75a9',
}
# Exact identities from the completed all-nine core package uploaded on 2026-10-25.
# No external score is read by this registry or used to select seeds/checkpoints.
FROZEN_REFERENCE = {'metadata_sha256': {'singleinput_plan.json': '9a82d523e1b6831b0331fcc00b387aa20bd6b52f813280a123a6c51ce92a73a6', 'queue_plan.json': 'b7881f35db213b7ef3b67463dd75d840134db2bbd242041f3d6fe7cef121c7da', 'evaluation_plan.json': '1b0571c32ce5d4e1225f28b741c47a2e890a3eed08863fba6e9185848bcb10a5', 'singleinput_status.json': '5c9d84a2477fab5fd537d21c56c772f15867068514164371ee59499aa629bf41'}, 'runs': {'raw_only/litemamba/seed_42': {'queue_requested_config.json': '926a12dad0fcbe6c347c3f889ff2cffc9f7edeb0a1425568980c2f9a6068e700', 'resolved_config.json': '5ea353ab98dc473a6602e8d780e15275665b757f7a19614e5f5a5f14771b9850', 'model_config.json': 'a5dde50a529bb8ef2539206637d6a92bd10781596197029dc6a2bdf732dae725', 'results.json': 'cdd629bf7d712fb8b2fa25d81115c0bc38e0c246fbb67bb64849bcb7b45121b8', 'calibration.json': '4d81f766d2851e3f39d46e950b58f5c4cde3ceb708cfd7776d1077dbf892cdac', 'epoch_metrics.jsonl': '49b98ec50c40e86f40998c0853ebe883fcffe15ddafa16f497a6e60fe9fcecef', 'val/analysis/metrics.json': '42e0a9442149e5ce139095aed9b2e43637b62471e4334edcaecff11113279e0f', 'environment.json': 'f57897c835b2b43819404b674650d524e8bab865784c7d7251bf9edcc9311b25', 'data_audit.json': 'ce013113204c675994b12421b4adec1d99fafa33f4e3af3ab0aec3a87f9088c3', 'source_snapshot/manifest.json': 'c66429791089e11eaa5ed9797bf3aa35bb62ad5adeb6d88b29595f59e394dc0e'}, 'raw_only/litemamba/seed_2024': {'queue_requested_config.json': '1650874adc5a106cd20366223290fddcdf3b907ab7cb35e2595748ecf9e0bf5e', 'resolved_config.json': '727ecd8399834e7bf821ac15572b4b4f7c2ee7268500d61e25e98321ca7c4d15', 'model_config.json': 'a5dde50a529bb8ef2539206637d6a92bd10781596197029dc6a2bdf732dae725', 'results.json': '70c425f9af62afc806d3f1833ba87f9524d33be92f4fd78ff9cea18c62e74941', 'calibration.json': '23c8f924ca03c5850877fea7a0888784c01483eba29f6084cb5c3a80353ee71c', 'epoch_metrics.jsonl': 'a3af4136bb8aeee1de2ab00d730e9df437eda5a597802cec03437b632a5897f7', 'val/analysis/metrics.json': '957edf4b20cb45d2fa67fb2ae47d7e7530ed7cf42d8edaba1aa620842542ceaa', 'environment.json': 'e931d21f512b543ee90f6ae0a0d6ef565cd431d53df7734bfe53a927398613b7', 'data_audit.json': '70316f551164ed5dd8455670b063bf3698b6531e3f6c926fb56250b5c3041b57', 'source_snapshot/manifest.json': 'c66429791089e11eaa5ed9797bf3aa35bb62ad5adeb6d88b29595f59e394dc0e'}, 'raw_only/litemamba/seed_2025': {'queue_requested_config.json': '38b9e7b22474c998457e134ac8da1be312235a24e35b10c20cae365a645a9a15', 'resolved_config.json': 'facb3d6a282af75f3759f6335d1157580b1887c39d76e34e2403d6fe34d49a1a', 'model_config.json': 'a5dde50a529bb8ef2539206637d6a92bd10781596197029dc6a2bdf732dae725', 'results.json': 'ecc867848250456b70309c4a6287cb93affa96a0ca49bcb9a76db714d6dae8d5', 'calibration.json': '797b78757fc00d7955e3c073fc18d0e623ce9b6153bf6196ff4d9e40aaec04cc', 'epoch_metrics.jsonl': 'f8a923e0c81b941451dae7c3b4635b2e91cc4038da1aa2f045f4951fb15ee2da', 'val/analysis/metrics.json': '1b8ef75942a81d85024f95bd0c17597b7c47aaab70fa81b755a45e22e49a3b71', 'environment.json': 'adb4612ab569d9d9c1acc4e8eed32480137be3b047540ebc2b0ed0c4ac590224', 'data_audit.json': 'd655e42dd54a6c51f33ff8f3e407f303e9bb33801da31084d485e9149a26e103', 'source_snapshot/manifest.json': 'c66429791089e11eaa5ed9797bf3aa35bb62ad5adeb6d88b29595f59e394dc0e'}, 'context_only/litemamba/seed_42': {'queue_requested_config.json': '1bdfd4194c1179fed1b568a398fd93fa5f6ea7198ad06b33152b158d2b2ec334', 'resolved_config.json': '3ed7f7e12e5842416c24b880c13e2d1bb1950fd6fe21de9715a59a76d9f9ab7b', 'model_config.json': '6d5bed7515b5b5777ba743f27f0ea755873cf4883e5a98f4b406bf1c59eecae6', 'results.json': '24ab10a73bc6ed353e536322617c36cdd4894635b66c19e9989641e5be333463', 'calibration.json': 'c89e88f78d820eaae046fe84a750ac23ae1bda784b4750111cba26e4bf670ac2', 'epoch_metrics.jsonl': 'fe9edd2973ec01af50fea2320b49b0ae8a13ba075a2a2fd03f381acb60e5e37b', 'val/analysis/metrics.json': '3fcdd356c3aaa0201d50afbfde34197d015c656f462eacaf364569f1711358f5', 'environment.json': '7f3c51804cd3d9fabbc94036ebf5d77eb2de50e95a3a99bdbfb9ea646ee841ae', 'data_audit.json': 'd10b0acc4a2ec28cf6f4e74f183ef7b176a46f79d0855899c20d530e3d213c85', 'source_snapshot/manifest.json': 'c66429791089e11eaa5ed9797bf3aa35bb62ad5adeb6d88b29595f59e394dc0e'}, 'context_only/litemamba/seed_2024': {'queue_requested_config.json': '076e7213fba67728b1eaf81ab408668d809b01b2dbd3b71c2c9de222e58f9085', 'resolved_config.json': '5e0a9ba0aa9d308349067c98311556d90e3e75baa06efd94e69add963ca34620', 'model_config.json': '6d5bed7515b5b5777ba743f27f0ea755873cf4883e5a98f4b406bf1c59eecae6', 'results.json': '0d55d6ca343236cc795e6eb0654ba45a5f4d7c326fbbbeed4d1f7455763b3d04', 'calibration.json': 'b396866cf3bcc1ebea57d0af385222ef277b2792d7bec8374772cb7a3cbc5c6c', 'epoch_metrics.jsonl': '5e5666c28f85b6e601459c8dd91fd884b45f8dfe8d8aa2d5d37f54ed57b4adcd', 'val/analysis/metrics.json': '65a942ba04e7cd660cab25df18cfb2f2d34245d12bc0cb057f0b9c3fe4764452', 'environment.json': 'a4e30a46d75200b605fb09143d163e2d96c436b85951d23ebc4128e84af0c9a4', 'data_audit.json': '3d0ce138f44d85e84ae3af37fbed17650ce130648ae162125a919b89eaa8a9f8', 'source_snapshot/manifest.json': 'c66429791089e11eaa5ed9797bf3aa35bb62ad5adeb6d88b29595f59e394dc0e'}, 'context_only/litemamba/seed_2025': {'queue_requested_config.json': 'a0002e232c40e9f251456a8750225c39f33fe1559838c6ff6d02ead21c438e61', 'resolved_config.json': 'bfd7b01ce9e0a604cdb08b3d7c400b4c16abeaf2deea05e051ac7ab667a95255', 'model_config.json': '6d5bed7515b5b5777ba743f27f0ea755873cf4883e5a98f4b406bf1c59eecae6', 'results.json': 'c2c8e8271a04b82ded2e56e007081915866ffa3d735891892394229adf7cf6ea', 'calibration.json': '566abd16ddfeef1b3843b9e89ed6680eb9d79b677eb227af4303aafbc4985964', 'epoch_metrics.jsonl': '9eacbd7ebe17bb5a50bfa2fc37be0216fa0248237eba54b9e32df623c1392326', 'val/analysis/metrics.json': '03ef5dce0b9d5501935d6a2f300cef559fc39c632457f991ba094f95b1a057c1', 'environment.json': 'b6d9e32f4b796198e755eb2523b2d77c4e5dc6dde29d009e1cf703deab0b84ce', 'data_audit.json': '8c4d6fd8fdb026dc0fe8c5224f3567abf3f6ed78bb261f159cdaf390229dbfac', 'source_snapshot/manifest.json': 'c66429791089e11eaa5ed9797bf3aa35bb62ad5adeb6d88b29595f59e394dc0e'}, 'responses_only/litemamba/seed_42': {'queue_requested_config.json': 'd56b4e313a4fd88f927029d7a364a008d6953b0d70d4e823d632fed2520a181f', 'resolved_config.json': '6ea22b0add0938efd685df4901ec36f92ef3bc6f1cd295b888cb5a1630e502de', 'model_config.json': '608421dde585281da6a9a71e91e017fe64f4b017aa8f2ff6bfcb08b1d10358ad', 'results.json': 'abb409545d6338e9dd88b0372dc04e9f0287d5229dbd6617285c84c13047324b', 'calibration.json': 'a972d8fdec2c0ff100811cd820efce246887dd72963fd7200e530bad57ce50e9', 'epoch_metrics.jsonl': '75c92ecc638b1f44009333cf6b0a6526169a3bde34793ae69a796e33abec165f', 'val/analysis/metrics.json': '30692c950ca0c382461a7c873c1f468253950d7c954c22483dffc1698da8a50d', 'environment.json': '85bc245d861e8e49cece58744172dd68b996c790b2cf87b33d435f55fe8350ba', 'data_audit.json': '1ecf1341ed5ec7946515a315a85c6e043fe7e41643c113b20e9866de110c7855', 'source_snapshot/manifest.json': 'c66429791089e11eaa5ed9797bf3aa35bb62ad5adeb6d88b29595f59e394dc0e'}, 'responses_only/litemamba/seed_2024': {'queue_requested_config.json': 'e39f8b46ab23131a74bbd828155692c891e8f3c6b51b968151f0aebe049c8c14', 'resolved_config.json': '9a00478f0a9a3b51959277be3c9b3e843131de0c1c3b0249c78c22c05ec5b707', 'model_config.json': '608421dde585281da6a9a71e91e017fe64f4b017aa8f2ff6bfcb08b1d10358ad', 'results.json': 'b1420578ca90f29d81d0d0f6e3d9511a0e90f629fdeed117168dddcdf8be4a23', 'calibration.json': '6664faa8363861fe1d179977119fe58dda648df05afda03a8471b84c72ae099c', 'epoch_metrics.jsonl': '33f73770dd6c0278385f9fbf47e2ca3d87262d76cacc56a942e67ca36ceb3746', 'val/analysis/metrics.json': '289474cd392de5aef21785638aab2300c02e0a0de5e7b3b157c4421edf34843b', 'environment.json': '34d97a394eddc77c14498127faf8dcae97fb0f131c3c95bc22dc103c81c8131a', 'data_audit.json': '5135b75248cc65df14dbbc44a6c95a487cff6a711a98a784ba4be537f1854507', 'source_snapshot/manifest.json': 'c66429791089e11eaa5ed9797bf3aa35bb62ad5adeb6d88b29595f59e394dc0e'}, 'responses_only/litemamba/seed_2025': {'queue_requested_config.json': '9191d0cee13d7726b6a1a5f0e7d0e8de7991ae2127781858ae8255d040165470', 'resolved_config.json': 'bceb8a321e5887e73c95b1dd2bc99507e53d0066717d818a718c3047177633fe', 'model_config.json': '608421dde585281da6a9a71e91e017fe64f4b017aa8f2ff6bfcb08b1d10358ad', 'results.json': 'f8b5c623ab4e50c29cfd024d100eaa71c251b374053d76afdc50a21cc09bf92c', 'calibration.json': '64a55ea52f3933bab84b45cd67bd67b31ff68d4a41ae2402cd461097bc86870e', 'epoch_metrics.jsonl': '61f572fa372151e96fbe69a9b9a5f59833282c4d3ecc05935e554c25d0050e20', 'val/analysis/metrics.json': '44cd5326176a84a09ecc711a3f3eec2cbbbaed5a450013d27222452105581f04', 'environment.json': 'bd1f8d0bc7e3849ff2d41b1e0f39ea9cca8c1d23778e924a296441a3da03a17a', 'data_audit.json': 'fd8a1fc9312423c099329c3cdaa75ea401530f19c96527ad42d3c08a2232c1b0', 'source_snapshot/manifest.json': 'c66429791089e11eaa5ed9797bf3aa35bb62ad5adeb6d88b29595f59e394dc0e'}}, 'checkpoints': {'raw_only/litemamba/seed_42': '625451b3047a1f8a2d25245d664b21624d27ba23fd7a6ca52eb11a89871ec13d', 'raw_only/litemamba/seed_2024': '33a2a0d0e8f3bb104f5352bdcbf8c21c675bb5daf4c5cd61cc32c1039b18d4d4', 'raw_only/litemamba/seed_2025': '67e144c8bc906843e6ec2af2d486d52940427e6105dfa43aeb5a85564dea3ea1', 'context_only/litemamba/seed_42': '3e72130d50985ebe88c7b10a8566f61403218c7e31f3ef20371dadbcfc16b0ed', 'context_only/litemamba/seed_2024': '2b13bef0b439a996396a43a60ff9437c851b070b039bdc74649697ec414501ce', 'context_only/litemamba/seed_2025': 'e0f8ba8a931867fde096359ce53079c80528c98e66e100ace21b67ee2a3431dd', 'responses_only/litemamba/seed_42': '146c8aabe1783391354a7e3f3829a0cabd1e7285db0a670e62dda9923bb251f4', 'responses_only/litemamba/seed_2024': '406221786f9477a2d9295e2ca6213488615aef34f53c6b75253247e8ecf855a1', 'responses_only/litemamba/seed_2025': '6ae2eeca7e3f792224a6353f938cf88dfb41351a03758d9a33aec85105b69534'}}


def _fail(condition, message):
    if not condition:
        raise ValueError(message)


def content(value):
    if isinstance(value, dict):
        if 'sha256' in value:
            return {'sha256':value['sha256'],'size_bytes':value.get('size_bytes',value.get('size'))}
        return {k:content(v) for k,v in value.items()}
    if isinstance(value,list):
        return [content(v) for v in value]
    return value


def _classes(value):
    return [value.get(str(i),value.get(i)) for i in range(6)] if isinstance(value,dict) else value


def training_sources(root):
    root=Path(root).expanduser().resolve()
    files={str(root/name):sha256(root/name) for name in FROZEN_REFERENCE['metadata_sha256']}
    files[str(root/'queue_manifest.json')]=sha256(root/'queue_manifest.json')
    return files


def _validate_training(root):
    for name,expected in FROZEN_REFERENCE['metadata_sha256'].items():
        _fail(sha256(root/name)==expected,'单输入训练冻结元数据与已上传完成版不同：'+str(root/name))
    plan=read_json(root/'singleinput_plan.json')
    freeze=read_json(root/'evaluation_plan.json')
    queue=read_json(root/'queue_manifest.json')
    status=read_json(root/'singleinput_status.json')
    _fail(plan.get('version')=='1.0.0' and plan.get('seeds')==list(SEEDS) and plan.get('schemes')==list(SCHEMES),'单输入训练计划身份不符')
    _fail(status.get('status')=='complete' and status.get('new_training_runs')==9 and status.get('internal_test_complete') is True,'单输入九次训练/测试尚未完整完成')
    _fail(queue.get('status')=='complete' and not queue.get('failed_runs') and queue.get('policy')=='train_all_defer_test_then_freeze_all_then_evaluate','九次训练队列未完整完成或冻结流程不同')
    _fail(freeze.get('policy')=='all_predeclared_training_succeeded_before_any_test;validation_only_selection','测试冻结策略不符')
    expected={f'{scheme}/litemamba/seed_{seed}' for scheme in SCHEMES for seed in SEEDS}
    for label,rows in [('冻结',freeze.get('experiments',[])),('队列',queue.get('experiments',[]))]:
        ids=[row.get('run_id') for row in rows]
        _fail(len(ids)==9 and set(ids)==expected,label+'必须完整包含预设九次运行')
    return plan,freeze


def _selection(directory, config, result):
    history=[json.loads(line) for line in (directory/'epoch_metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    _fail(history and [x['epoch'] for x in history]==list(range(1,len(history)+1)),'训练历史不连续')
    _fail(all(x.get('monitor','val_loss')=='val_loss' and math.isfinite(float(x['val']['loss'])) for x in history),'验证选模规则或损失无效')
    best=min(history,key=lambda row:float(row['val']['loss']))
    _fail(best['epoch']==result.get('best_epoch'),'不是validation loss所选检查点')
    metrics=read_json(directory/'val/analysis/metrics.json')
    _fail(_classes(metrics.get('class_definition'))==CLASSES,'验证指标类别定义不同')
    macro=float(metrics['overall']['raw']['macro_f1'])
    _fail(abs(float(best['val']['macro_f1'])-macro)<1e-9,'最佳轮次和验证导出不一致')
    _fail(config.get('checkpoint') is None and config.get('monitor')=='val_loss' and not config.get('external_manifest'),'训练初始化或选模规则与既定方案不同')
    return int(best['epoch'])


def discover_singleinput_runs(training_dir, previous_plan):
    """Return (nine runs, preprocessing), never infer, train or inspect external scores."""
    root=Path(training_dir).expanduser().resolve()
    plan,freeze=_validate_training(root)
    reference=[r for r in previous_plan.get('runs',[]) if r.get('model')==MODEL and r.get('feature_scheme')=='full']
    _fail(len(reference)==3 and {r.get('seed') for r in reference}==set(SEEDS),'旧Figure4缺少三个完整输入 GatedConv，不能用SSM替代')
    base=reference[0]
    preprocessing=copy.deepcopy(previous_plan['preprocessing'])
    for name,expected in RUNTIME_SHA256.items():
        _fail(sha256(VENDOR/name)==expected,'本包冻结推理源码不同：'+name)
    frozen={x['run_id']:x for x in freeze['experiments']}
    selections=training_sources(root)
    runs=[]
    for scheme in SCHEMES:
        for seed in SEEDS:
            training_id=f'{scheme}/litemamba/seed_{seed}'
            directory=root/training_id
            metadata=FROZEN_REFERENCE['runs'][training_id]
            for name,expected in metadata.items():
                _fail(sha256(directory/name)==expected,'训练记录不同于已上传九次完成版：'+str(directory/name))
            config_path=directory/'resolved_config.json'
            calibration_path=directory/'calibration.json'
            config=read_json(config_path)
            requested=read_json(directory/'queue_requested_config.json')
            model_config=read_json(directory/'model_config.json')
            result=read_json(directory/'results.json')
            calibration=read_json(calibration_path)
            kwargs=copy.deepcopy(config['model_kwargs'])
            for current in (config,requested,result):
                _fail((current.get('model_type'),current.get('seed'),current.get('feature_scheme'))==('litemamba',seed,scheme),'目录、模型、特征方案或种子身份不符')
                _fail(current.get('litemamba_variant')=='legacy_gatedconv' and current.get('readout')=='target_attention','必须使用最终GatedConv与第4位读出')
            for current in (config,requested):
                _fail(content(current['input_fingerprints'])==content(base['training_input_fingerprints']),'单输入与full的训练/验证/测试数据内容不同')
            _fail(result.get('status')=='complete' and result.get('test_evaluated') is True,'单输入运行未完成')
            _fail(_classes(config.get('class_names'))==CLASSES and _classes(model_config.get('class_names'))==CLASSES,'类别顺序必须为A,T,C,G,X,Y')
            _fail(config.get('target_index')==model_config.get('target_index')==3 and model_config.get('seq_len')==6 and model_config.get('num_classes')==6,'只接受第4位为N的新六分类模型')
            _fail(model_config.get('one_hot_base_order')==['A','C','T','G','N'],'序列one-hot顺序不同')
            _fail(all(kwargs.get(k)==v for k,v in {'feature_scheme':scheme,'litemamba_variant':'legacy_gatedconv','readout':'target_attention','target_index':3}.items()),'model_kwargs特征掩码或读出配置不同')
            full_kwargs=dict(base['model_kwargs']);full_kwargs['feature_scheme']=scheme
            _fail(kwargs==full_kwargs,'单输入除了输入掩码以外的模型参数与full不同')
            for k in ('feature_scheme','readout','litemamba_variant'):
                _fail(model_config.get(k)==config.get(k),'model_config与实际配置不同：'+k)
            for k in ('raw_signal_len','dwell_divisor','quality_divisor'):
                _fail(float(config[k])==float(preprocessing[k]),'输入预处理不同：'+k)
            source=config['source_snapshot']['files_sha256']
            _fail(all(source.get(k)==v for k,v in RUNTIME_SHA256.items()),'实际训练模型/预处理源码不同')
            checkpoint=directory/'best_checkpoint.pt'
            digest=sha256(checkpoint)
            _fail(digest==FROZEN_REFERENCE['checkpoints'][training_id]==frozen[training_id]['checkpoint_sha256']==calibration.get('checkpoint_sha256')==result.get('provenance',{}).get('checkpoint_sha256'),'checkpoint、温度、结果或训练冻结SHA不一致：'+training_id)
            temperature=float(calibration['temperature'])
            _fail(calibration.get('fitted_on')=='val' and math.isfinite(temperature) and temperature>0,'校准必须仅用validation拟合正温度')
            epoch=_selection(directory,config,result)
            run={'run_id':f'{scheme}/{MODEL}/seed_{seed}','model':MODEL,'actual_model_type':'litemamba','structure_experiment':MODEL,
                 'training_run_id':training_id,'seed':seed,'feature_scheme':scheme,'run_dir':str(directory),'checkpoint':str(checkpoint),
                 'checkpoint_sha256':digest,'resolved_config':str(config_path),'resolved_config_sha256':sha256(config_path),
                 'calibration':str(calibration_path),'calibration_sha256':sha256(calibration_path),'temperature':temperature,
                 'model_kwargs':kwargs,'runtime_dir':str(VENDOR),'runtime_sha256':dict(RUNTIME_SHA256),
                 'training_input_fingerprints':config['input_fingerprints'],'training_source_sha256':source,
                 'frozen_metadata_sha256':dict(metadata),'selection_sources':dict(selections),'selected_epoch':epoch,
                 'profile_key':identity_hash({'model':MODEL,'kwargs':kwargs,'preprocessing':preprocessing})[:20]}
            verify_run(run)
            runs.append(run)
    return runs,preprocessing
