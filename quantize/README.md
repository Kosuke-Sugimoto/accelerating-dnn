## Memorandum
### torch.cuda.synchronize()について
CPUのみの経過時間計測なら`time.time()`を利用すれば問題ないが、GPUを利用する場合にはGPUが非同期処理を行っていることを念頭に入れなければならない。  
その場合に利用するのが`torch.cuda.synchronize()`で、機能としては非同期処理の結果が返ってくるのを待つというもの。  
これらを時間計測初めと終わりの手前に配置することで正確なGPUでの計算時間が計測できる。  
（カーネル側のコードにも似たような`synchronize()`系のやつあった気がする…）  
誤った手法で計測した時間を含む結果は以下の通り
```shell
root@777ac295d16c:/work/quantize# python half_precision.py 
FP32 running time: 0.27257585525512695
FP16 running time: 0.15159344673156738
Wrong FP32 running time: 0.0016565322875976562
Wrong FP16 running time: 0.002227783203125
```
参考文献
- https://www.mattari-benkyo-note.com/2021/03/21/pytorch-cuda-time-measurement/
