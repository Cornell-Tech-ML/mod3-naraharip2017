# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

### Module 3.2 Parallel Script Output
```
Parallel loop listing for  Function _tensor_matrix_multiply, /Users/pavan/Documents/Weill Cornell/Fall 2024/Machine Learning Engineering/workspace/mod3-naraharip2017/minitorch/fast_ops.py (294)
------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                              |
    out: Storage,                                                                         |
    out_shape: Shape,                                                                     |
    out_strides: Strides,                                                                 |
    a_storage: Storage,                                                                   |
    a_shape: Shape,                                                                       |
    a_strides: Strides,                                                                   |
    b_storage: Storage,                                                                   |
    b_shape: Shape,                                                                       |
    b_strides: Strides,                                                                   |
) -> None:                                                                                |
    """NUMBA tensor matrix multiply function.                                             |
                                                                                          |
    Should work for any tensor shapes that broadcast as long as                           |
                                                                                          |
    ```                                                                                   |
    assert a_shape[-1] == b_shape[-2]                                                     |
    ```                                                                                   |
                                                                                          |
    Optimizations:                                                                        |
                                                                                          |
    * Outer loop in parallel                                                              |
    * No index buffers or function calls                                                  |
    * Inner loop should have no global writes, 1 multiply.                                |
                                                                                          |
                                                                                          |
    Args:                                                                                 |
    ----                                                                                  |
        out (Storage): storage for `out` tensor                                           |
        out_shape (Shape): shape for `out` tensor                                         |
        out_strides (Strides): strides for `out` tensor                                   |
        a_storage (Storage): storage for `a` tensor                                       |
        a_shape (Shape): shape for `a` tensor                                             |
        a_strides (Strides): strides for `a` tensor                                       |
        b_storage (Storage): storage for `b` tensor                                       |
        b_shape (Shape): shape for `b` tensor                                             |
        b_strides (Strides): strides for `b` tensor                                       |
                                                                                          |
    Returns:                                                                              |
    -------                                                                               |
        None : Fills in `out`                                                             |
                                                                                          |
    """                                                                                   |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                |
                                                                                          |
    num_batches = a_shape[0] if a_batch_stride > 0 else 1                                 |
    num_rows_a = a_shape[-2]                                                              |
    num_cols_a_rows_b = a_shape[-1]                                                       |
    num_cols_b = b_shape[-1]                                                              |
                                                                                          |
    for batch in prange(num_batches):-----------------------------------------------------| #9
        a_start = batch * a_batch_stride if a_batch_stride > 0 else 0                     |
        b_start = batch * b_batch_stride if b_batch_stride > 0 else 0                     |
                                                                                          |
        for i in range(num_rows_a):                                                       |
            for j in range(num_cols_b):                                                   |
                dot_product = 0.0                                                         |
                for k in range(num_cols_a_rows_b):                                        |
                    a_pos = a_start + i * a_strides[-2] + k * a_strides[-1]               |
                    b_pos = b_start + k * b_strides[-2] + j * b_strides[-1]               |
                    dot_product += a_storage[a_pos] * b_storage[b_pos]                    |
                                                                                          |
                out_pos = (                                                               |
                    batch * out_strides[0] + i * out_strides[-2] + j * out_strides[-1]    |
                )                                                                         |
                out[out_pos] = dot_product                                                |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #9).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

# Training

## CPU

### Simple Data Set

```
!cd $DIR; PYTHONPATH=/content/$DIR python3.10 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
```

```
Epoch  0  | Loss  4.006263764553608  | Correct 44 | Time Per Epoch 16.022907733917236
Epoch  10  | Loss  0.8280258254689707  | Correct 49 | Time Per Epoch 1.9552662372589111
Epoch  20  | Loss  1.2845685927086377  | Correct 50 | Time Per Epoch 1.0998747462318057
Epoch  30  | Loss  1.252373190888954  | Correct 49 | Time Per Epoch 0.7971069043682467
Epoch  40  | Loss  0.779503409507094  | Correct 49 | Time Per Epoch 0.6413328124255668
Epoch  50  | Loss  0.6918769411476393  | Correct 50 | Time Per Epoch 0.5536588640773997
Epoch  60  | Loss  0.046336529705463265  | Correct 50 | Time Per Epoch 0.5044934710518258
Epoch  70  | Loss  0.27367867451141975  | Correct 50 | Time Per Epoch 0.4556390466824384
Epoch  80  | Loss  0.37573237382924546  | Correct 49 | Time Per Epoch 0.4192016625110014
Epoch  90  | Loss  0.8962786798349481  | Correct 50 | Time Per Epoch 0.3904338773790297
Epoch  100  | Loss  0.7073567406204986  | Correct 50 | Time Per Epoch 0.3675377180080603
Epoch  110  | Loss  0.3965072947785613  | Correct 50 | Time Per Epoch 0.3486189004537222
Epoch  120  | Loss  0.6919138126112079  | Correct 50 | Time Per Epoch 0.3354020631017764
Epoch  130  | Loss  0.8452408236205807  | Correct 50 | Time Per Epoch 0.32875506386502096
Epoch  140  | Loss  0.5566063734986377  | Correct 49 | Time Per Epoch 0.316634457162086
Epoch  150  | Loss  0.6543417682835281  | Correct 50 | Time Per Epoch 0.3062105241990247
Epoch  160  | Loss  0.2599602191341485  | Correct 50 | Time Per Epoch 0.29702866299552205
Epoch  170  | Loss  0.16657336320660018  | Correct 50 | Time Per Epoch 0.28893032826875387
Epoch  180  | Loss  0.14308326678483926  | Correct 50 | Time Per Epoch 0.281756146836676
Epoch  190  | Loss  0.042424288675680384  | Correct 50 | Time Per Epoch 0.27668725008739853
Epoch  200  | Loss  0.2442328877252674  | Correct 50 | Time Per Epoch 0.2751969176145335
Epoch  210  | Loss  0.03935816794219296  | Correct 50 | Time Per Epoch 0.2699275468763017
Epoch  220  | Loss  0.39712540521251594  | Correct 50 | Time Per Epoch 0.264997766028702
Epoch  230  | Loss  0.452251139062689  | Correct 50 | Time Per Epoch 0.2603249611792626
Epoch  240  | Loss  0.0009369292085309826  | Correct 50 | Time Per Epoch 0.2561684525359221
Epoch  250  | Loss  0.06740416286461882  | Correct 50 | Time Per Epoch 0.2522216983050464
Epoch  260  | Loss  0.0004258310213459709  | Correct 50 | Time Per Epoch 0.24925503785582795
Epoch  270  | Loss  0.13835951387452386  | Correct 50 | Time Per Epoch 0.249575289413058
Epoch  280  | Loss  0.13287541462142405  | Correct 50 | Time Per Epoch 0.24631597478194592
Epoch  290  | Loss  0.0742351260300309  | Correct 50 | Time Per Epoch 0.24325913006497413
Epoch  300  | Loss  0.07519494348679748  | Correct 50 | Time Per Epoch 0.24044497068538223
Epoch  310  | Loss  0.12009572263613627  | Correct 50 | Time Per Epoch 0.2377936472080145
Epoch  320  | Loss  0.25093826414493287  | Correct 50 | Time Per Epoch 0.23529872121840623
Epoch  330  | Loss  0.24991826690444494  | Correct 50 | Time Per Epoch 0.23307197979929944
Epoch  340  | Loss  0.06454195703361078  | Correct 50 | Time Per Epoch 0.2341210275801046
Epoch  350  | Loss  0.032606380715763675  | Correct 50 | Time Per Epoch 0.23196337161920008
Epoch  360  | Loss  0.03291884520066866  | Correct 50 | Time Per Epoch 0.2298939029926078
Epoch  370  | Loss  0.06978866896722434  | Correct 50 | Time Per Epoch 0.2279690950707284
Epoch  380  | Loss  0.10252317432159345  | Correct 50 | Time Per Epoch 0.2261109452235104
Epoch  390  | Loss  0.014962111918624746  | Correct 50 | Time Per Epoch 0.22450888065425942
Epoch  400  | Loss  0.08730548298346101  | Correct 50 | Time Per Epoch 0.22286376988798604
Epoch  410  | Loss  0.020794356410087006  | Correct 50 | Time Per Epoch 0.22429040806716957
Epoch  420  | Loss  0.10992051605  | Correct 50 | Time Per Epoch 0.22281832819596606
Epoch  430  | Loss  0.06147584263524464  | Correct 50 | Time Per Epoch 0.22133958256714859
Epoch  440  | Loss  0.1032991037734667  | Correct 50 | Time Per Epoch 0.2199533331691543
Epoch  450  | Loss  0.02545466974980556  | Correct 50 | Time Per Epoch 0.2185772263554406
Epoch  460  | Loss  0.21355866247691369  | Correct 50 | Time Per Epoch 0.21730089549645942
Epoch  470  | Loss  0.06328550341699414  | Correct 50 | Time Per Epoch 0.21606573785186572
Epoch  480  | Loss  8.011216019140162e-05  | Correct 50 | Time Per Epoch 0.21729190780814125
Epoch  490  | Loss  0.2184610297393107  | Correct 50 | Time Per Epoch 0.21611299145731566
```

### Split Data Set

```
!cd $DIR; PYTHONPATH=/content/$DIR python3.10 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
```

```
Epoch  0  | Loss  4.172733784902132  | Correct 32 | Time Per Epoch 16.18531036376953
Epoch  10  | Loss  3.962782979141677  | Correct 45 | Time Per Epoch 1.9216314879330723
Epoch  20  | Loss  2.696725585729261  | Correct 47 | Time Per Epoch 1.0819077832358224
Epoch  30  | Loss  2.4628611207970748  | Correct 47 | Time Per Epoch 0.7845205491588961
Epoch  40  | Loss  1.9658225902848991  | Correct 48 | Time Per Epoch 0.6435143017187351
Epoch  50  | Loss  4.07102149216851  | Correct 47 | Time Per Epoch 0.5620977644826851
Epoch  60  | Loss  4.086546546990794  | Correct 46 | Time Per Epoch 0.5170788217763431
Epoch  70  | Loss  1.8460642190191234  | Correct 48 | Time Per Epoch 0.4668349548124931
Epoch  80  | Loss  1.5715038875026375  | Correct 48 | Time Per Epoch 0.42893658449620375
Epoch  90  | Loss  1.333609228731721  | Correct 48 | Time Per Epoch 0.3992958121247344
Epoch  100  | Loss  2.4603845462361  | Correct 48 | Time Per Epoch 0.376581078708762
Epoch  110  | Loss  0.8405050348988463  | Correct 49 | Time Per Epoch 0.3663973077997431
Epoch  120  | Loss  0.6141247403531732  | Correct 49 | Time Per Epoch 0.34946422931576565
Epoch  130  | Loss  1.4965371669257477  | Correct 48 | Time Per Epoch 0.334973236986699
Epoch  140  | Loss  1.2760417542971036  | Correct 49 | Time Per Epoch 0.3224858493669659
Epoch  150  | Loss  0.586018003472336  | Correct 49 | Time Per Epoch 0.31163569317748213
Epoch  160  | Loss  0.23037224217833804  | Correct 49 | Time Per Epoch 0.302115601782473
Epoch  170  | Loss  0.4462956261968105  | Correct 49 | Time Per Epoch 0.29438224870559065
Epoch  180  | Loss  1.8788964997208717  | Correct 50 | Time Per Epoch 0.2926057109516629
Epoch  190  | Loss  0.25410385716981226  | Correct 50 | Time Per Epoch 0.28553715925566187
Epoch  200  | Loss  0.14057589049933747  | Correct 49 | Time Per Epoch 0.2792745075415616
Epoch  210  | Loss  1.6021001738039542  | Correct 49 | Time Per Epoch 0.2735028278206197
Epoch  220  | Loss  0.8360653030914529  | Correct 49 | Time Per Epoch 0.26831940827865947
Epoch  230  | Loss  0.9505022575225998  | Correct 49 | Time Per Epoch 0.2635871135827267
Epoch  240  | Loss  0.7838275745425903  | Correct 49 | Time Per Epoch 0.25933474821668445
Epoch  250  | Loss  1.7838477113750015  | Correct 50 | Time Per Epoch 0.2597176335247389
Epoch  260  | Loss  0.8894872359947636  | Correct 49 | Time Per Epoch 0.2558389431672078
Epoch  270  | Loss  1.5912784898788797  | Correct 49 | Time Per Epoch 0.25224188509022616
Epoch  280  | Loss  0.44805034258231957  | Correct 49 | Time Per Epoch 0.24892134937951574
Epoch  290  | Loss  0.20469477869484215  | Correct 50 | Time Per Epoch 0.24593809462085212
Epoch  300  | Loss  0.500024087360997  | Correct 49 | Time Per Epoch 0.24301662635169552
Epoch  310  | Loss  0.7459700920310991  | Correct 49 | Time Per Epoch 0.24029587549411982
Epoch  320  | Loss  0.4375081830637192  | Correct 49 | Time Per Epoch 0.24125492090005368
Epoch  330  | Loss  0.5364615235707788  | Correct 49 | Time Per Epoch 0.23878696245729383
Epoch  340  | Loss  1.3022471092794021  | Correct 50 | Time Per Epoch 0.2364540505618993
Epoch  350  | Loss  0.6391814928406705  | Correct 49 | Time Per Epoch 0.2342460291338103
Epoch  360  | Loss  0.6938128028431378  | Correct 49 | Time Per Epoch 0.2321665003028933
Epoch  370  | Loss  0.08737784776535096  | Correct 50 | Time Per Epoch 0.23017160461919328
Epoch  380  | Loss  0.1399819807955497  | Correct 49 | Time Per Epoch 0.2283154447560548
Epoch  390  | Loss  0.09246196240780857  | Correct 50 | Time Per Epoch 0.22943064379875008
Epoch  400  | Loss  0.12478657405674143  | Correct 50 | Time Per Epoch 0.22769885764752243
Epoch  410  | Loss  0.6048513518684147  | Correct 49 | Time Per Epoch 0.22606032608199295
Epoch  420  | Loss  0.2020904494072625  | Correct 49 | Time Per Epoch 0.22442811628418693
Epoch  430  | Loss  0.2639105871247751  | Correct 50 | Time Per Epoch 0.2229223516866945
Epoch  440  | Loss  0.7465957581039607  | Correct 49 | Time Per Epoch 0.22143844412027303
Epoch  450  | Loss  0.620281218198636  | Correct 49 | Time Per Epoch 0.22011076161708112
Epoch  460  | Loss  1.8690393506706517  | Correct 49 | Time Per Epoch 0.22143882008798726
Epoch  470  | Loss  0.8138294105561213  | Correct 49 | Time Per Epoch 0.2201134940874298
Epoch  480  | Loss  0.04053259778367482  | Correct 50 | Time Per Epoch 0.2188551544895291
Epoch  490  | Loss  1.5300977700946996  | Correct 50 | Time Per Epoch 0.2176225185394287
```

### Xor Data Set

```
!cd $DIR; PYTHONPATH=/content/$DIR python3.10 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
```

```
Epoch  0  | Loss  7.618228608243477  | Correct 29 | Time Per Epoch 16.086260557174683
Epoch  10  | Loss  5.032478755287994  | Correct 45 | Time Per Epoch 1.9239851344715466
Epoch  20  | Loss  4.803121750306605  | Correct 46 | Time Per Epoch 1.1341193744114466
Epoch  30  | Loss  3.6659428805522842  | Correct 42 | Time Per Epoch 0.8196989336321431
Epoch  40  | Loss  2.722808359488156  | Correct 47 | Time Per Epoch 0.6585447148578923
Epoch  50  | Loss  3.6762395801853445  | Correct 46 | Time Per Epoch 0.560477051080442
Epoch  60  | Loss  3.388958645197291  | Correct 44 | Time Per Epoch 0.4961655530773225
Epoch  70  | Loss  2.06115835807969  | Correct 44 | Time Per Epoch 0.4486284692522506
Epoch  80  | Loss  2.6886800894728746  | Correct 44 | Time Per Epoch 0.4146558679180381
Epoch  90  | Loss  2.568284876776182  | Correct 46 | Time Per Epoch 0.39731775535331976
Epoch  100  | Loss  4.316778771860042  | Correct 41 | Time Per Epoch 0.37368221094112586
Epoch  110  | Loss  2.536703711205413  | Correct 46 | Time Per Epoch 0.3542089419321971
Epoch  120  | Loss  1.690125117961862  | Correct 49 | Time Per Epoch 0.33813814289313704
Epoch  130  | Loss  3.431111556046999  | Correct 47 | Time Per Epoch 0.3245797794283801
Epoch  140  | Loss  2.953961208202586  | Correct 49 | Time Per Epoch 0.3130146899121873
Epoch  150  | Loss  2.5309125364097307  | Correct 49 | Time Per Epoch 0.3034577827579928
Epoch  160  | Loss  1.0483304355393444  | Correct 48 | Time Per Epoch 0.301124215866468
Epoch  170  | Loss  1.1983048560516096  | Correct 49 | Time Per Epoch 0.292795879799023
Epoch  180  | Loss  1.4210300788843697  | Correct 49 | Time Per Epoch 0.2853476671882756
Epoch  190  | Loss  0.7009289020001209  | Correct 49 | Time Per Epoch 0.278706328407008
Epoch  200  | Loss  1.2742742885488936  | Correct 49 | Time Per Epoch 0.27270313519150463
Epoch  210  | Loss  0.991021378288532  | Correct 49 | Time Per Epoch 0.2673345154495601
Epoch  220  | Loss  1.1040902416969043  | Correct 49 | Time Per Epoch 0.26245166597323183
Epoch  230  | Loss  0.3604685276039698  | Correct 49 | Time Per Epoch 0.2627901834842963
Epoch  240  | Loss  0.8043091462011702  | Correct 49 | Time Per Epoch 0.2585037902183058
Epoch  250  | Loss  1.1298460239601142  | Correct 49 | Time Per Epoch 0.25447208187969556
Epoch  260  | Loss  1.3597622409738679  | Correct 49 | Time Per Epoch 0.2508103171527614
Epoch  270  | Loss  2.4146929679557365  | Correct 49 | Time Per Epoch 0.24735062500647514
Epoch  280  | Loss  1.0737177937879254  | Correct 49 | Time Per Epoch 0.2442533435346393
Epoch  290  | Loss  1.0844948218933026  | Correct 49 | Time Per Epoch 0.24131923278992118
Epoch  300  | Loss  1.799734642150869  | Correct 48 | Time Per Epoch 0.24242162070797132
Epoch  310  | Loss  0.6569004269019683  | Correct 49 | Time Per Epoch 0.23982445066764807
Epoch  320  | Loss  1.096452131340974  | Correct 49 | Time Per Epoch 0.23729111249573134
Epoch  330  | Loss  1.2081726007411508  | Correct 50 | Time Per Epoch 0.23503216083676434
Epoch  340  | Loss  0.9419682908188716  | Correct 48 | Time Per Epoch 0.2327648425731491
Epoch  350  | Loss  0.5930866294512929  | Correct 50 | Time Per Epoch 0.23070838920071593
Epoch  360  | Loss  0.6168657841993519  | Correct 49 | Time Per Epoch 0.22872074423074062
Epoch  370  | Loss  0.6107926829746732  | Correct 49 | Time Per Epoch 0.22998133374031662
Epoch  380  | Loss  0.23301900978013396  | Correct 50 | Time Per Epoch 0.22807801927481425
Epoch  390  | Loss  0.5537860486105839  | Correct 49 | Time Per Epoch 0.22631694532721244
Epoch  400  | Loss  0.975797492445307  | Correct 50 | Time Per Epoch 0.22462653043561445
Epoch  410  | Loss  0.20456166633344888  | Correct 49 | Time Per Epoch 0.22303458431921447
Epoch  420  | Loss  1.2305438912057383  | Correct 49 | Time Per Epoch 0.2215056170193996
Epoch  430  | Loss  0.7033072190123568  | Correct 50 | Time Per Epoch 0.2200274174285322
Epoch  440  | Loss  0.46895377905617236  | Correct 49 | Time Per Epoch 0.22124412973451507
Epoch  450  | Loss  0.47834444714495283  | Correct 50 | Time Per Epoch 0.21985532914984252
Epoch  460  | Loss  0.47023910319740353  | Correct 50 | Time Per Epoch 0.2185784275774842
Epoch  470  | Loss  0.14508290050289707  | Correct 49 | Time Per Epoch 0.2173212865355668
Epoch  480  | Loss  0.425319429956869  | Correct 49 | Time Per Epoch 0.21614382073685928
Epoch  490  | Loss  1.8162251184676907  | Correct 49 | Time Per Epoch 0.21503603288452397
```

## GPU

### Simple

```
!cd $DIR; PYTHONPATH=/content/$DIR python3.10 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
```

```
Epoch  0  | Loss  6.987210167762221  | Correct 42 | Time Per Epoch 4.570101022720337
Epoch  10  | Loss  3.3807445911965575  | Correct 47 | Time Per Epoch 2.2745510881597344
Epoch  20  | Loss  0.6063045780416515  | Correct 48 | Time Per Epoch 2.108073972520374
Epoch  30  | Loss  2.161993635110011  | Correct 50 | Time Per Epoch 2.061309891362344
Epoch  40  | Loss  0.737148894050382  | Correct 49 | Time Per Epoch 2.042868683977825
Epoch  50  | Loss  1.1632650673919014  | Correct 50 | Time Per Epoch 2.015191452175963
Epoch  60  | Loss  0.584751305891838  | Correct 50 | Time Per Epoch 2.0083072654536513
Epoch  70  | Loss  0.4199752993083645  | Correct 50 | Time Per Epoch 1.994323364445861
Epoch  80  | Loss  0.77055722649059  | Correct 50 | Time Per Epoch 1.9933640750837915
Epoch  90  | Loss  0.4815309783978223  | Correct 50 | Time Per Epoch 1.992066917838631
Epoch  100  | Loss  0.6387583694691217  | Correct 50 | Time Per Epoch 1.9909774501725
Epoch  110  | Loss  0.7097974203085379  | Correct 49 | Time Per Epoch 1.9896106569616645
Epoch  120  | Loss  0.3095582122014131  | Correct 50 | Time Per Epoch 1.9829205442066036
Epoch  130  | Loss  1.573620388413728  | Correct 49 | Time Per Epoch 1.9837393815280826
Epoch  140  | Loss  0.16869312578920703  | Correct 50 | Time Per Epoch 1.9839540153530473
Epoch  150  | Loss  0.10719934078944221  | Correct 50 | Time Per Epoch 1.9789085498708763
Epoch  160  | Loss  0.10794244121844043  | Correct 50 | Time Per Epoch 1.9782666330752166
Epoch  170  | Loss  0.6251800674797547  | Correct 50 | Time Per Epoch 1.973364998722634
Epoch  180  | Loss  0.32705694660347384  | Correct 50 | Time Per Epoch 1.9735277291819535
Epoch  190  | Loss  0.43235895724547985  | Correct 50 | Time Per Epoch 1.9736821476701667
Epoch  200  | Loss  0.4073748456119126  | Correct 50 | Time Per Epoch 1.9694162530092458
Epoch  210  | Loss  0.05660932604879007  | Correct 50 | Time Per Epoch 1.9696293363073991
Epoch  220  | Loss  0.9794340302788219  | Correct 50 | Time Per Epoch 1.9699766538801236
Epoch  230  | Loss  0.06180831532327379  | Correct 50 | Time Per Epoch 1.9697122326144925
Epoch  240  | Loss  0.0743496367075983  | Correct 50 | Time Per Epoch 1.970054495878734
Epoch  250  | Loss  0.7727130698620138  | Correct 50 | Time Per Epoch 1.9672665852474502
```

