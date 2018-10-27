# readTensorflowGraphFromPy
c++にpython埋め込んで画像を食わせてtensorflowのグラフを動かすサンプル。

# 環境
OS:windows 10  
MSVC:v140(v141未確認)  
pytnon 3.5.2::anaconda4.2.0(64bit)  
tensorflow(1.3.0)

# how to use
pythonのdllがreleaseしかないのでdebugでは実行できません。
debugで実行したければpythonのdllを自分でビルドするなどの闇が必要です。

## include 
opencvのincとanacondaのincを通す
```
C:\*\openCV3.2.0(x64)\install\include
C:\Users\*\Anaconda3\include
```

## lib
pythonのパスとopencvのパスを通す
```
C:\*\openCV3.2.0(x64)\install\x64\vc14\lib
C:\Users\*\Anaconda3\libs
```

# 実行
release x64でビルド。ビルド先にpython/に入っているスクリプトと.pbをぶち込む。
でcmdとかで
```
load_tensor_graph.exe 画像ファイル名
```
で実行すると推論結果が得られる。
