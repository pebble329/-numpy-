#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy
class cn_array(numpy.ndarray):
	def __new__(cls, input_array, info=None):
		obj = numpy.asarray(input_array).view(cls)
		return obj
	def __array_finalize__(self, obj):
        
		if obj is None: return
	'''




	#!/usr/bin/python
	# -*- coding: UTF-8 -*-
	import numpy
	class cn_array(numpy.ndarray):
		def __new__(cls, input_array, info=None):
			obj = numpy.asarray(input_array).view(cls)
			return obj
		def __array_finalize__(self, obj):
			if obj is None: return
	'''
	def 一序列(形状,  数据类型=None,  顺序='C'):
		return cn_array(numpy.ones(shape = 形状,  dtype = 数据类型,  order = 顺序))

	#def 一序列(形状,  数据类型=None,  顺序='C'):
	#	return cn_array(numpy.ones(shape = 形状,  dtype = 数据类型,  order = 顺序))
	def AxisError(轴,  维度=None,  msg_prefix=None):
		return cn_array(numpy.AxisError(axis = 轴,  ndim = 维度,  msg_prefix = msg_prefix))

	def DataSource(destpath='.'):
		return cn_array(numpy.DataSource(destpath = destpath))


	def Tester(package=None,  raise_warnings='release',  depth=0,  check_fpu_mode=False):
		return cn_array(numpy.Tester(package = package,  raise_warnings = raise_warnings,  depth = depth,  check_fpu_mode = check_fpu_mode))


	def 添加新文档(地点,  对象,  文本,  编译器警告=True):
		return cn_array(numpy.add_newdoc(place = 地点,  obj = 对象,  doc = 文本,  warn_on_python = 编译器警告))

	def 序列长度(序列):
		return cn_array(numpy.alen(a = 序列))

	def 全部(序列,  轴=None,  输出=None,  保留维度集=numpy._NoValue):
		return cn_array(numpy.all(a = 序列,  axis = 轴,  out = 输出,  keepdims = 保留维度集))

	def 全序列近似(序列,  b,  相对公差=1e-05,  绝对公差=1e-08,  空值相等=False):
		return cn_array(numpy.allclose(a = 序列,  b = b,  rtol = 相对公差,  atol = 绝对公差,  equal_nan = 空值相等))

	def 全部真(*args,  **kwargs):
		return cn_array(numpy.alltrue(*args,  **kwargs))

	def 序列最大值(序列,  轴=None,  输出=None,  保留维度集=numpy._NoValue,  初始值=numpy._NoValue,  条件=numpy._NoValue):
		return cn_array(numpy.amax(a = 序列,  axis = 轴,  out = 输出,  keepdims = 保留维度集,  initial = 初始值,  where = 条件))

	def 序列最小值(序列,  轴=None,  输出=None,  保留维度集=numpy._NoValue,  初始值=numpy._NoValue,  条件=numpy._NoValue):
		return cn_array(numpy.amin(a = 序列,  axis = 轴,  out = 输出,  keepdims = 保留维度集,  initial = 初始值,  where = 条件))

	def 角度(z,  度=False):
		return cn_array(numpy.angle(z = z,  deg = 度))

	def 所有(序列,  轴=None,  输出=None,  保留维度集=numpy._NoValue):
		return cn_array(numpy.any(a = 序列,  axis = 轴,  out = 输出,  keepdims = 保留维度集))

	def 追加(序列,  值集,  轴=None):
		return cn_array(numpy.append(arr = 序列,  values = 值集,  axis = 轴))

	def 沿轴应用(函数1维,  轴,  序列,  *args,  **kwargs):
		return cn_array(numpy.apply_along_axis(func1d = 函数1维,  axis = 轴,  arr = 序列,  *args,  **kwargs))

	def 在轴上应用(功能,  序列,  轴数):
		return cn_array(numpy.apply_over_axes(func = 功能,  a = 序列,  axes = 轴数))

	def 索引序列最大值(序列,  轴=None,  输出=None):
		return cn_array(numpy.argmax(a = 序列,  axis = 轴,  out = 输出))

	def 索引序列最小值(序列,  轴=None,  输出=None):
		return cn_array(numpy.argmin(a = 序列,  axis = 轴,  out = 输出))

	def 索引分割(序列,  第几个,  轴=-1,  种类='introselect',  顺序=None):
		return cn_array(numpy.argpartition(a = 序列,  kth = 第几个,  axis = 轴,  kind = 种类,  order = 顺序))

	def 索引排序(序列,  轴=-1,  种类=None,  顺序=None):
		return cn_array(numpy.argsort(a = 序列,  axis = 轴,  kind = 种类,  order = 顺序))

	def 索引条件(序列):
		return cn_array(numpy.argwhere(a = 序列))

	def 序列四舍五入(序列,  小数点=0,  输出=None):
		return cn_array(numpy.around(a = 序列,  decimals = 小数点,  out = 输出))

	def 序列转字符串(序列,  最大线宽=None,  精度=None,  抑制_极小值=None,  分隔符=' ',  字首='',  样式=numpy._NoValue,  formatter=None,  阈值=None,  边缘条目=None,  符号=None,  floatmode=None,  suffix='',  **kwarg):
		return cn_array(numpy.array2string(a = 序列,  max_line_width = 最大线宽,  precision = 精度,  suppress_small = 抑制_极小值,  separator = 分隔符,  prefix = 字首,  style = 样式,  formatter = formatter,  threshold = 阈值,  edgeitems = 边缘条目,  sign = 符号,  floatmode = floatmode,  suffix = suffix,  **kwarg))

	def 序列相等(a1,  a2):
		return cn_array(numpy.array_equal(a1 = a1,  a2 = a2))

	def 序列等价(a1,  a2):
		return cn_array(numpy.array_equiv(a1 = a1,  a2 = a2))

	def 序列格式化(序列,  最大线宽=None,  精度=None,  抑制_极小值=None):
		return cn_array(numpy.array_repr(arr = 序列,  max_line_width = 最大线宽,  precision = 精度,  suppress_small = 抑制_极小值))

	def 序列拆分(序列,  切片集或分块集,  轴=0):
		return cn_array(numpy.array_split(ary = 序列,  indices_or_sections = 切片集或分块集,  axis = 轴))

	def 序列_字符串(序列,  最大线宽=None,  精度=None,  抑制_极小值=None):
		return cn_array(numpy.array_str(a = 序列,  max_line_width = 最大线宽,  precision = 精度,  suppress_small = 抑制_极小值))

	def 转为全部序列(序列,  数据类型=None,  顺序=None):
		return cn_array(numpy.asanyarray(a = 序列,  dtype = 数据类型,  order = 顺序))

	def 转为序列(序列,  数据类型=None,  顺序=None):
		return cn_array(numpy.asarray(a = 序列,  dtype = 数据类型,  order = 顺序))

	def 转为序列_检测限制(序列,  数据类型=None,  顺序=None):
		return cn_array(numpy.asarray_chkfinite(a = 序列,  dtype = 数据类型,  order = 顺序))

	def 转为连续存储函数(序列,  数据类型=None):
		return cn_array(numpy.ascontiguousarray(a = 序列,  dtype = 数据类型))

	def 转为浮点序列(序列,  数据类型= numpy.float64 ):
		return cn_array(numpy.asfarray(a = 序列,  dtype = 数据类型))

	def 转为fortran序列(序列,  数据类型=None):
		return cn_array(numpy.asfortranarray(a = 序列,  dtype = 数据类型))

	def 转为矩阵(数据,  数据类型=None):
		return cn_array(numpy.asmatrix(data = 数据,  dtype = 数据类型))

	def 转为标量(序列):
		return cn_array(numpy.asscalar(a = 序列))

	def 至少一维(*arys):
		return cn_array(numpy.atleast_1d(*arys))

	def 至少二维(*arys):
		return cn_array(numpy.atleast_2d(*arys))

	def 至少三维(*arys):
		return cn_array(numpy.atleast_3d(*arys))

	def 加权平均数(序列,  轴=None,  权重=None,  回到=False):
		return cn_array(numpy.average(a = 序列,  axis = 轴,  weights = 权重,  returned = 回到))

	def 巴特利特窗(M):
		return cn_array(numpy.bartlett(M = M))

	def n进制格式化(数,  基础=2,  填充=0):
		return cn_array(numpy.base_repr(number = 数,  base = 基础,  padding = 填充))

	def 二进制格式化(数,  宽度=None):
		return cn_array(numpy.binary_repr(num = 数,  width = 宽度))

	def 布莱克曼窗(M):
		return cn_array(numpy.blackman(M = M))

	def 块拼接(序列集):
		return cn_array(numpy.block(arrays = 序列集))

	def 矩阵拼接(对象,  局部字典=None,  全局字典=None):
		return cn_array(numpy.bmat(obj = 对象,  ldict = 局部字典,  gdict = 全局字典))

	def 广播序列(*args,  **kwargs):
		return cn_array(numpy.broadcast_arrays(*args,  **kwargs))

	def 广播到(序列,  形状,  包含子类=False):
		return cn_array(numpy.broadcast_to(array = 序列,  shape = 形状,  subok = 包含子类))

	def 字节界限(序列):
		return cn_array(numpy.byte_bounds(a = 序列))

	def 双精度复数(实数=0,  虚部=0):
		return cn_array(numpy.cdouble(real = 实数,  imag = 虚部))

	def 浮点复数(实数=0,  虚部=0):
		return cn_array(numpy.cfloat(real = 实数,  imag = 虚部))

	def 字符():
		return cn_array(numpy.character())

	def 字符列表(形状,  条目大小=1,  统一编码=False,  缓冲=None,  偏移集=0,  步伐=None,  顺序='C'):
		return cn_array(numpy.chararray(shape = 形状,  itemsize = 条目大小,  unicode = 统一编码,  buffer = 缓冲,  offset = 偏移集,  strides = 步伐,  order = 顺序))

	def 挑选(序列,  选择,  输出=None,  模式='raise'):
		return cn_array(numpy.choose(a = 序列,  choices = 选择,  out = 输出,  mode = 模式))

	def 夹逼(序列,  序列最小值,  序列最大值,  输出=None,  **kwargs):
		return cn_array(numpy.clip(a = 序列,  a_min = 序列最小值,  a_max = 序列最大值,  out = 输出,  **kwargs))

	def 列堆叠(元组):
		return cn_array(numpy.column_stack(tup = 元组))

	def 普通类型(*序列集):
		return cn_array(numpy.common_type(*arrays))

	#ef complex(实数=0,  虚部=0):
	#return cn_array(numpy.complex(real = 实数,  imag = 虚部))

	def 复数128位(实数=0,  虚部=0):
		return cn_array(numpy.complex128(real = 实数,  imag = 虚部))

	def 复数_(实数=0,  虚部=0):
		return cn_array(numpy.complex_(real = 实数,  imag = 虚部))

	def 浮点型复数():
		return cn_array(numpy.complexfloating())

	def 打包(条件,  序列,  轴=None,  输出=None):
		return cn_array(numpy.compress(condition = 条件,  a = 序列,  axis = 轴,  out = 输出))

	def 卷积(序列,  v,  模式='full'):
		return cn_array(numpy.convolve(a = 序列,  v = v,  mode = 模式))

	def 复制(序列,  顺序='K'):
		return cn_array(numpy.copy(a = 序列,  order = 顺序))

	def 相关系数(x,  y=None,  行变量=True,  偏向=numpy._NoValue,  自由度=numpy._NoValue):
		return cn_array(numpy.corrcoef(x = x,  y = y,  rowvar = 行变量,  bias = 偏向,  ddof = 自由度))

	def 正则协方差(序列,  v,  模式='valid'):
		return cn_array(numpy.correlate(a = 序列,  v = v,  mode = 模式))

	def 非零计数(序列,  轴=None):
		return cn_array(numpy.count_nonzero(a = 序列,  axis = 轴))

	def 协方差(m,  y=None,  行变量=True,  偏向=False,  自由度=None,  频率权重=None,  序列权重=None):
		return cn_array(numpy.cov(m = m,  y = y,  rowvar = 行变量,  bias = 偏向,  ddof = 自由度,  fweights = 频率权重,  aweights = 序列权重))

	def 叉积(序列,  b,  轴a=-1,  轴b=-1,  轴c=-1,  轴=None):
		return cn_array(numpy.cross(a = 序列,  b = b,  axisa = 轴a,  axisb = 轴b,  axisc = 轴c,  axis = 轴))

	def 累乘(序列,  轴=None,  数据类型=None,  输出=None):
		return cn_array(numpy.cumprod(a = 序列,  axis = 轴,  dtype = 数据类型,  out = 输出))

	def 累乘法(*args,  **kwargs):
		return cn_array(numpy.cumproduct(*args,  **kwargs))

	def 累加(序列,  轴=None,  数据类型=None,  输出=None):
		return cn_array(numpy.cumsum(a = 序列,  axis = 轴,  dtype = 数据类型,  out = 输出))

	def 删除(序列,  对象,  轴=None):
		return cn_array(numpy.delete(arr = 序列,  obj = 对象,  axis = 轴))

	def 别名警示(*args,  **kwargs):
		return cn_array(numpy.deprecate(*args,  **kwargs))

	def 别名文档警示(信息):
		return cn_array(numpy.deprecate_with_doc(msg = 信息))

	def 对角线(v,  k=0):
		return cn_array(numpy.diag(v = v,  k = k))

	def 对角线索引(n,  维度=2):
		return cn_array(numpy.diag_indices(n = n,  ndim = 维度))

	def 生成对角线索引(序列):
		return cn_array(numpy.diag_indices_from(arr = 序列))

	def 对角线铺平(v,  k=0):
		return cn_array(numpy.diagflat(v = v,  k = k))

	def 对角线变换(序列,  偏移集=0,  轴1=0,  轴2=1):
		return cn_array(numpy.diagonal(a = 序列,  offset = 偏移集,  axis1 = 轴1,  axis2 = 轴2))

	def 项差(序列,  n=1,  轴=-1,  前置=numpy._NoValue,  附加=numpy._NoValue):
		return cn_array(numpy.diff(a = 序列,  n = n,  axis = 轴,  prepend = 前置,  append = 附加))

	def 数字化(x,  块数,  右边=False):
		return cn_array(numpy.digitize(x = x,  bins = 块数,  right = 右边))

	def 显示(消息,  设备=None,  换行=True):
		return cn_array(numpy.disp(mesg = 消息,  device = 设备,  linefeed = 换行))

	def 双精度数(x):
		return cn_array(numpy.double(x = x))

	def 维度拆分(序列,  切片集或分块集):
		return cn_array(numpy.dsplit(ary = 序列,  indices_or_sections = 切片集或分块集))

	def 维度堆叠(元组):
		return cn_array(numpy.dstack(tup = 元组))

	def 一维项差(序列,  到尾=None,  到头=None):
		return cn_array(numpy.ediff1d(ary = 序列,  to_end = 到尾,  to_begin = 到头))

	def 范式求和(*operands,  **kwargs):
		return cn_array(numpy.einsum(*operands,  **kwargs))

	def 范式路径(*operands,  **kwargs):
		return cn_array(numpy.einsum_path(*operands,  **kwargs))

	def 错误状态(**kwargs):
		return cn_array(numpy.errstate(**kwargs))

	def 扩维(序列,  轴):
		return cn_array(numpy.expand_dims(a = 序列,  axis = 轴))

	def 提取(条件,  序列):
		return cn_array(numpy.extract(condition = 条件,  arr = 序列))

	def 类单位序列(N,  M=None,  k=0,  数据类型=numpy.float,  顺序='C'):
		return cn_array(numpy.eye(N = N,  M = M,  k = k,  dtype = 数据类型,  order = 顺序))

	def 对角线填充(序列,  值,  打包=False):
		return cn_array(numpy.fill_diagonal(a = 序列,  val = 值,  wrap = 打包))

	def 查找常用类型(序列类型集,  标量类型):
		return cn_array(numpy.find_common_type(array_types = 序列类型集,  scalar_types = 标量类型))

	def 浮点变量类型信息(数据类型):
		return cn_array(numpy.finfo(dtype = 数据类型))

	def 向0取整(x,  输出=None):
		return cn_array(numpy.fix(x = x,  out = 输出))

	def 铺平器():
		return cn_array(numpy.flatiter())

	def 铺平非零数索引(序列):
		return cn_array(numpy.flatnonzero(a = 序列))

	def 可变数据类型():
		return cn_array(numpy.flexible())

	def 翻转(m,  轴=None):
		return cn_array(numpy.flip(m = m,  axis = 轴))

	def 左右翻转(m):
		return cn_array(numpy.fliplr(m = m))

	def 上下翻转(m):
		return cn_array(numpy.flipud(m = m))

	#ef float(x):
	#return cn_array(numpy.float(x = x))

	def 浮点数64位(x):
		return cn_array(numpy.float64(x = x))

	def 浮点数_(x):
		return cn_array(numpy.float_(x = x))

	def 浮点型():
		return cn_array(numpy.floating())

	def 定位浮点格式(x,  精度=None,  唯一=True,  小数=True,  修边='k',  符号=False,  填充左边距=None,  pad_right=None):
		return cn_array(numpy.format_float_positional(x = x,  precision = 精度,  unique = 唯一,  fractional = 小数,  trim = 修边,  sign = 符号,  pad_left = 填充左边距,  pad_right = pad_right))

	def 科学计算浮点格式(x,  精度=None,  唯一=True,  修边='k',  符号=False,  填充左边距=None,  算式小数位=None):
		return cn_array(numpy.format_float_scientific(x = x,  precision = 精度,  unique = 唯一,  trim = 修边,  sign = 符号,  pad_left = 填充左边距,  exp_digits = 算式小数位))

	def 格式分析器(格式集,  名字集,  标题集,  是否对齐=False,  字节顺序=None):
		return cn_array(numpy.format_parser(formats = 格式集,  names = 名字集,  titles = 标题集,  aligned = 是否对齐,  byteorder = 字节顺序))

	def 函数生成序列(函数,  形状,  **kwargs):
		return cn_array(numpy.fromfunction(function = 函数,  shape = 形状,  **kwargs))

	def 正则生成(文件,  正则表达式,  数据类型,  编码方式=None):
		return cn_array(numpy.fromregex(file = 文件,  regexp = 正则表达式,  dtype = 数据类型,  encoding = 编码方式))

	def 常量序列(形状,  填充值,  数据类型=None,  顺序='C'):
		return cn_array(numpy.full(shape = 形状,  fill_value = 填充值,  dtype = 数据类型,  order = 顺序))

	def 同形常量序列(序列,  填充值,  数据类型=None,  顺序='K',  包含子类=True,  形状=None):
		return cn_array(numpy.full_like(a = 序列,  fill_value = 填充值,  dtype = 数据类型,  order = 顺序,  subok = 包含子类,  shape = 形状))

	def 未来价值(利率,  期数,  支付值,  现在价值,  时间点='end'):
		return cn_array(numpy.fv(rate = 利率,  nper = 期数,  pmt = 支付值,  pv = 现在价值,  when = 时间点))

	def 通用数据类型():
		return cn_array(numpy.generic())

	def 文本生成(文件名,  数据类型=numpy.float,  注释='#',  定界符=None,  跳过头部=0,  跳过尾部=0,  转换器=None,  missing_values=None,  filling_values=None,  使用列=None,  名字集=None,  excludelist=None,  deletechars=" !#$%&'()*+,-./:;<=>?@[\\]^{|}~",  replace_space='_',  autostrip=False,  case_sensitive=True,  defaultfmt='f%i',  unpack=None,  usemask=False,  loose=True,  invalid_raise=True,  max_rows=None,  编码方式='bytes'):
		return cn_array(numpy.genfromtxt(fname = 文件名,  dtype = 数据类型,  comments = 注释,  delimiter = 定界符,  skip_header = 跳过头部,  skip_footer = 跳过尾部,  converters = 转换器,  missing_values = missing_values,  filling_values = filling_values,  usecols = 使用列,  names = 名字集,  excludelist = excludelist,  deletechars = deletechars,  replace_space = replace_space,  autostrip = autostrip,  case_sensitive = case_sensitive,  defaultfmt = defaultfmt,  unpack = unpack,  usemask = usemask,  loose = loose,  invalid_raise = invalid_raise,  max_rows = max_rows,  encoding = 编码方式))

	def 地理空间(开始,  停止,  数=50,  终点=True,  数据类型=None,  轴=0):
		return cn_array(numpy.geomspace(start = 开始,  stop = 停止,  num = 数,  endpoint = 终点,  dtype = 数据类型,  axis = 轴))

	def 获取序列包装(*args):
		return cn_array(numpy.get_array_wrap(*args))

	def 获取引用文件():
		return cn_array(numpy.get_include())

	def 获取打印选项():
		return cn_array(numpy.get_printoptions())

	def 获取缓冲大小():
		return cn_array(numpy.getbufsize())

	def 获取错误():
		return cn_array(numpy.geterr())

	def 获取错误回调函数():
		return cn_array(numpy.geterrcall())

	def 梯度(f,  *varargs,  **kwargs):
		return cn_array(numpy.gradient(f = f,  *varargs,  **kwargs))

	def 汉明窗(M):
		return cn_array(numpy.hamming(M = M))

	def 汉宁窗(M):
		return cn_array(numpy.hanning(M = M))

	def 区间分布(序列,  块数=10,  范围=None,  规范=None,  权重=None,  目的=None):
		return cn_array(numpy.histogram(a = 序列,  bins = 块数,  range = 范围,  normed = 规范,  weights = 权重,  density = 目的))

	def 二维区间分布(x,  y,  块数=10,  范围=None,  规范=None,  权重=None,  目的=None):
		return cn_array(numpy.histogram2d(x = x,  y = y,  bins = 块数,  range = 范围,  normed = 规范,  weights = 权重,  density = 目的))

	def 区间分布_分界点(序列,  块数=10,  范围=None,  权重=None):
		return cn_array(numpy.histogram_bin_edges(a = 序列,  bins = 块数,  range = 范围,  weights = 权重))

	def 多维区间分布(样品,  块数=10,  范围=None,  规范=None,  权重=None,  目的=None):
		return cn_array(numpy.histogramdd(sample = 样品,  bins = 块数,  range = 范围,  normed = 规范,  weights = 权重,  density = 目的))

	def 水平拆分(序列,  切片集或分块集):
		return cn_array(numpy.hsplit(ary = 序列,  indices_or_sections = 切片集或分块集))

	def 水平堆叠(元组):
		return cn_array(numpy.hstack(tup = 元组))

	def 贝塞尔变形(x):
		return cn_array(numpy.i0(x = x))

	def 单位序列(n,  数据类型=None):
		return cn_array(numpy.identity(n = n,  dtype = 数据类型))

	def 整型变量类型信息(整数类型):
		return cn_array(numpy.iinfo(int_type = 整数类型))

	def 虚部(值):
		return cn_array(numpy.imag(val = 值))

	def 一维包含集(序列1,  序列2,  默认唯一值=False,  反转=False):
		return cn_array(numpy.in1d(ar1 = 序列1,  ar2 = 序列2,  assume_unique = 默认唯一值,  invert = 反转))

	def 索引序列(维度集合,  数据类型=numpy.int,  稀疏=False):
		return cn_array(numpy.indices(dimensions = 维度集合,  dtype = 数据类型,  sparse = 稀疏))

	def 不精确类型():
		return cn_array(numpy.inexact())

	def 信息(对象=None,  最大宽度=76,  输出=None,  顶层='numpy'):
		return cn_array(numpy.info(object = 对象,  maxwidth = 最大宽度,  output = 输出,  toplevel = 顶层))

	def 插入(序列,  对象,  值集,  轴=None):
		return cn_array(numpy.insert(arr = 序列,  obj = 对象,  values = 值集,  axis = 轴))

	def 整型():
		return cn_array(numpy.integer())

	def 插值(x,  xp,  文件指针,  左边=None,  右边=None,  期=None):
		return cn_array(numpy.interp(x = x,  xp = xp,  fp = 文件指针,  left = 左边,  right = 右边,  period = 期))

	def 一维交集(序列1,  序列2,  默认唯一值=False,  返回切片=False):
		return cn_array(numpy.intersect1d(ar1 = 序列1,  ar2 = 序列2,  assume_unique = 默认唯一值,  return_indices = 返回切片))

	def 每月归还利息(利率,  每一个,  期数,  现在价值,  未来价值=0,  时间点='end'):
		return cn_array(numpy.ipmt(rate = 利率,  per = 每一个,  nper = 期数,  pv = 现在价值,  fv = 未来价值,  when = 时间点))

	def 内部收益率(值集):
		return cn_array(numpy.irr(values = 值集))

	def 是否近似(序列,  b,  相对公差=1e-05,  绝对公差=1e-08,  空值相等=False):
		return cn_array(numpy.isclose(a = 序列,  b = b,  rtol = 相对公差,  atol = 绝对公差,  equal_nan = 空值相等))

	def 是否复数(x):
		return cn_array(numpy.iscomplex(x = x))

	def 是否复数对象(x):
		return cn_array(numpy.iscomplexobj(x = x))

	def 是否fortran(序列):
		return cn_array(numpy.isfortran(a = 序列))

	def 是否包含(元素,  测试元素集,  默认唯一值=False,  反转=False):
		return cn_array(numpy.isin(element = 元素,  test_elements = 测试元素集,  assume_unique = 默认唯一值,  invert = 反转))

	def 是否负无穷(x,  输出=None):
		return cn_array(numpy.isneginf(x = x,  out = 输出))

	def 是否正无穷(x,  输出=None):
		return cn_array(numpy.isposinf(x = x,  out = 输出))

	def 是否实数(x):
		return cn_array(numpy.isreal(x = x))

	def 以色列(x):
		return cn_array(numpy.isrealobj(x = x))

	def 是否标量(元素):
		return cn_array(numpy.isscalar(element = 元素))

	def 是否标量类型(代表):
		return cn_array(numpy.issctype(rep = 代表))

	def 是否子程序(参数1,  参数2):
		return cn_array(numpy.issubclass_(arg1 = 参数1,  arg2 = 参数2))

	def 是否属于类型(参数1,  参数2):
		return cn_array(numpy.issubdtype(arg1 = 参数1,  arg2 = 参数2))

	def 问题类型(参数1,  参数2):
		return cn_array(numpy.issubsctype(arg1 = 参数1,  arg2 = 参数2))

	def 可迭代(y):
		return cn_array(numpy.iterable(y = y))

	def 直积索引(*args):
		return cn_array(numpy.ix_(*args))

	def 凯泽窗(M,  贝塔):
		return cn_array(numpy.kaiser(M = M,  beta = 贝塔))

	def 克罗内克积(序列,  b):
		return cn_array(numpy.kron(a = 序列,  b = b))

	def 邻接空间(开始,  停止,  数=50,  终点=True,  返回步长=False,  数据类型=None,  轴=0):
		return cn_array(numpy.linspace(start = 开始,  stop = 停止,  num = 数,  endpoint = 终点,  retstep = 返回步长,  dtype = 数据类型,  axis = 轴))

	def 读取(文件,  读取模式=None,  是否可串行化=False,  固定导入=True,  编码方式='ASCII'):
		return cn_array(numpy.load(file = 文件,  mmap_mode = 读取模式,  allow_pickle = 是否可串行化,  fix_imports = 固定导入,  encoding = 编码方式))

	def 读取文件(*args,  **kwargs):
		return cn_array(numpy.loads(*args,  **kwargs))

	def 读取文本(文件名,  数据类型=numpy.float,  注释='#',  定界符=None,  转换器=None,  跳行=0,  使用列=None,  unpack=False,  维度极小值=0,  编码方式='bytes',  max_rows=None):
		return cn_array(numpy.loadtxt(fname = 文件名,  dtype = 数据类型,  comments = 注释,  delimiter = 定界符,  converters = 转换器,  skiprows = 跳行,  usecols = 使用列,  unpack = unpack,  ndmin = 维度极小值,  encoding = 编码方式,  max_rows = max_rows))

	def 对数空间(开始,  停止,  数=50,  终点=True,  基础=10.0,  数据类型=None,  轴=0):
		return cn_array(numpy.logspace(start = 开始,  stop = 停止,  num = 数,  endpoint = 终点,  base = 基础,  dtype = 数据类型,  axis = 轴))

	def 查询(内容,  模块=None,  引入模块=True,  再生=False,  输出=None):
		return cn_array(numpy.lookfor(what = 内容,  module = 模块,  import_modules = 引入模块,  regenerate = 再生,  output = 输出))

	def 文本生成遮罩(文件名,  **kwargs):
		return cn_array(numpy.mafromtxt(fname = 文件名,  **kwargs))

	def 遮罩索引(n,  遮罩函数,  k=0):
		return cn_array(numpy.mask_indices(n = n,  mask_func = 遮罩函数,  k = k))

	def 矩阵解析(数据,  数据类型=None):
		return cn_array(numpy.mat(data = 数据,  dtype = 数据类型))

	def 矩阵(数据,  数据类型=None,  复制=True):
		return cn_array(numpy.matrix(data = 数据,  dtype = 数据类型,  copy = 复制))

	def max(序列,  轴=None,  输出=None,  保留维度集=numpy._NoValue,  初始值=numpy._NoValue,  条件=numpy._NoValue):
		return cn_array(numpy.max(a = 序列,  axis = 轴,  out = 输出,  keepdims = 保留维度集,  initial = 初始值,  where = 条件))

	def 最大标量类型(t):
		return cn_array(numpy.maximum_sctype(t = t))

	def 平均值(序列,  轴=None,  数据类型=None,  输出=None,  保留维度集=numpy._NoValue):
		return cn_array(numpy.mean(a = 序列,  axis = 轴,  dtype = 数据类型,  out = 输出,  keepdims = 保留维度集))

	def 中值的(序列,  轴=None,  输出=None,  输入超限=False,  保留维度集=False):
		return cn_array(numpy.median(a = 序列,  axis = 轴,  out = 输出,  overwrite_input = 输入超限,  keepdims = 保留维度集))

	def 记忆地图(文件名,  数据类型= numpy.uint8 ,  模式='r+',  偏移集=0,  形状=None,  顺序='C'):
		return cn_array(numpy.memmap(filename = 文件名,  dtype = 数据类型,  mode = 模式,  offset = 偏移集,  shape = 形状,  order = 顺序))

	def 网格(*xi,  **kwargs):
		return cn_array(numpy.meshgrid(*xi,  **kwargs))

	def min(序列,  轴=None,  输出=None,  保留维度集=numpy._NoValue,  初始值=numpy._NoValue,  条件=numpy._NoValue):
		return cn_array(numpy.min(a = 序列,  axis = 轴,  out = 输出,  keepdims = 保留维度集,  initial = 初始值,  where = 条件))

	def 最小类型码(类型字符集,  类型集='GDFgdf',  默认='d'):
		return cn_array(numpy.mintypecode(typechars = 类型字符集,  typeset = 类型集,  default = 默认))

	def 修正内部收益率(值集,  经济利率,  再投资率):
		return cn_array(numpy.mirr(values = 值集,  finance_rate = 经济利率,  reinvest_rate = 再投资率))

	def 移动轴(序列,  来源,  目的地):
		return cn_array(numpy.moveaxis(a = 序列,  source = 来源,  destination = 目的地))

	def 主轴排序(序列):
		return cn_array(numpy.msort(a = 序列))

	def 空转0(x,  复制=True,  空值=0.0,  位置信息=None,  负数信息=None):
		return cn_array(numpy.nan_to_num(x = x,  copy = 复制,  nan = 空值,  posinf = 位置信息,  neginf = 负数信息))

	def 跳空索引最大值(序列,  轴=None):
		return cn_array(numpy.nanargmax(a = 序列,  axis = 轴))

	def 跳空索引最小值(序列,  轴=None):
		return cn_array(numpy.nanargmin(a = 序列,  axis = 轴))

	def 跳空累乘(序列,  轴=None,  数据类型=None,  输出=None):
		return cn_array(numpy.nancumprod(a = 序列,  axis = 轴,  dtype = 数据类型,  out = 输出))

	def 跳空累加(序列,  轴=None,  数据类型=None,  输出=None):
		return cn_array(numpy.nancumsum(a = 序列,  axis = 轴,  dtype = 数据类型,  out = 输出))

	def 跳空最大值(序列,  轴=None,  输出=None,  保留维度集=numpy._NoValue):
		return cn_array(numpy.nanmax(a = 序列,  axis = 轴,  out = 输出,  keepdims = 保留维度集))

	def 跳空平均值(序列,  轴=None,  数据类型=None,  输出=None,  保留维度集=numpy._NoValue):
		return cn_array(numpy.nanmean(a = 序列,  axis = 轴,  dtype = 数据类型,  out = 输出,  keepdims = 保留维度集))

	def 跳空中位数(序列,  轴=None,  输出=None,  输入超限=False,  保留维度集=numpy._NoValue):
		return cn_array(numpy.nanmedian(a = 序列,  axis = 轴,  out = 输出,  overwrite_input = 输入超限,  keepdims = 保留维度集))

	def 跳空最小值(序列,  轴=None,  输出=None,  保留维度集=numpy._NoValue):
		return cn_array(numpy.nanmin(a = 序列,  axis = 轴,  out = 输出,  keepdims = 保留维度集))

	def 跳空百分比(序列,  q,  轴=None,  输出=None,  输入超限=False,  插值='linear',  保留维度集=numpy._NoValue):
		return cn_array(numpy.nanpercentile(a = 序列,  q = q,  axis = 轴,  out = 输出,  overwrite_input = 输入超限,  interpolation = 插值,  keepdims = 保留维度集))

	def 跳空求积(序列,  轴=None,  数据类型=None,  输出=None,  保留维度集=numpy._NoValue):
		return cn_array(numpy.nanprod(a = 序列,  axis = 轴,  dtype = 数据类型,  out = 输出,  keepdims = 保留维度集))

	def 跳空分位数(序列,  q,  轴=None,  输出=None,  输入超限=False,  插值='linear',  保留维度集=numpy._NoValue):
		return cn_array(numpy.nanquantile(a = 序列,  q = q,  axis = 轴,  out = 输出,  overwrite_input = 输入超限,  interpolation = 插值,  keepdims = 保留维度集))

	def 跳空标准差(序列,  轴=None,  数据类型=None,  输出=None,  自由度=0,  保留维度集=numpy._NoValue):
		return cn_array(numpy.nanstd(a = 序列,  axis = 轴,  dtype = 数据类型,  out = 输出,  ddof = 自由度,  keepdims = 保留维度集))

	def 跳空求和(序列,  轴=None,  数据类型=None,  输出=None,  保留维度集=numpy._NoValue):
		return cn_array(numpy.nansum(a = 序列,  axis = 轴,  dtype = 数据类型,  out = 输出,  keepdims = 保留维度集))

	def 跳空方差(序列,  轴=None,  数据类型=None,  输出=None,  自由度=0,  保留维度集=numpy._NoValue):
		return cn_array(numpy.nanvar(a = 序列,  axis = 轴,  dtype = 数据类型,  out = 输出,  ddof = 自由度,  keepdims = 保留维度集))

	def 序列枚举(序列):
		return cn_array(numpy.ndenumerate(arr = 序列))

	def 文本生成序列(文件名,  **kwargs):
		return cn_array(numpy.ndfromtxt(fname = 文件名,  **kwargs))

	def 维度(序列):
		return cn_array(numpy.ndim(a = 序列))

	def 序列索引(*形状):
		return cn_array(numpy.ndindex(*shape))

	def 非零索引(序列):
		return cn_array(numpy.nonzero(a = 序列))

	def 分期数(利率,  支付值,  现在价值,  未来价值=0,  时间点='end'):
		return cn_array(numpy.nper(rate = 利率,  pmt = 支付值,  pv = 现在价值,  fv = 未来价值,  when = 时间点))

	def 净现值(利率,  值集):
		return cn_array(numpy.npv(rate = 利率,  values = 值集))

	def 数字类型():
		return cn_array(numpy.number())

	def 对象转标量(代表,  默认=None):
		return cn_array(numpy.obj2sctype(rep = 代表,  default = 默认))

	def object():
		return cn_array(numpy.object())

	#ef 一序列(形状,  数据类型=None,  顺序='C'):
	#return cn_array(numpy.ones(shape = 形状,  dtype = 数据类型,  order = 顺序))

	def 同形一序列(序列,  数据类型=None,  顺序='K',  包含子类=True,  形状=None):
		return cn_array(numpy.ones_like(a = 序列,  dtype = 数据类型,  order = 顺序,  subok = 包含子类,  shape = 形状))

	def 外积(序列,  b,  输出=None):
		return cn_array(numpy.outer(a = 序列,  b = b,  out = 输出))

	def 填充(序列,  填充宽度,  模式='constant',  **kwargs):
		return cn_array(numpy.pad(array = 序列,  pad_width = 填充宽度,  mode = 模式,  **kwargs))

	def 大小分区(序列,  第几个,  轴=-1,  种类='introselect',  顺序=None):
		return cn_array(numpy.partition(a = 序列,  kth = 第几个,  axis = 轴,  kind = 种类,  order = 顺序))

	def 百分位数(序列,  q,  轴=None,  输出=None,  输入超限=False,  插值='linear',  保留维度集=False):
		return cn_array(numpy.percentile(a = 序列,  q = q,  axis = 轴,  out = 输出,  overwrite_input = 输入超限,  interpolation = 插值,  keepdims = 保留维度集))

	def 分段计算(x,  条件列表,  功能列表,  *args,  **kw):
		return cn_array(numpy.piecewise(x = x,  condlist = 条件列表,  funclist = 功能列表,  *args,  **kw))

	def 放置(序列,  遮罩,  变量集):
		return cn_array(numpy.place(arr = 序列,  mask = 遮罩,  vals = 变量集))

	def 每月还贷(利率,  期数,  现在价值,  未来价值=0,  时间点='end'):
		return cn_array(numpy.pmt(rate = 利率,  nper = 期数,  pv = 现在价值,  fv = 未来价值,  when = 时间点))

	def 根求多项式(零序列):
		return cn_array(numpy.poly(seq_of_zeros = 零序列))

	def 一维多项式(c_or_r,  r=False,  变量=None):
		return cn_array(numpy.poly1d(c_or_r = c_or_r,  r = r,  variable = 变量))

	def 多项式求和(a1,  a2):
		return cn_array(numpy.polyadd(a1 = a1,  a2 = a2))

	def 多项式微分(p,  m=1):
		return cn_array(numpy.polyder(p = p,  m = m))

	def 多项式求商(u,  v):
		return cn_array(numpy.polydiv(u = u,  v = v))

	def 多项式拟合(x,  y,  度,  相对关系=None,  满=False,  w=None,  相关=False):
		return cn_array(numpy.polyfit(x = x,  y = y,  deg = 度,  rcond = 相对关系,  full = 满,  w = w,  cov = 相关))

	def 多项式积分(p,  m=1,  k=None):
		return cn_array(numpy.polyint(p = p,  m = m,  k = k))

	def 多项式求积(a1,  a2):
		return cn_array(numpy.polymul(a1 = a1,  a2 = a2))

	def 多项式求差(a1,  a2):
		return cn_array(numpy.polysub(a1 = a1,  a2 = a2))

	def 多项式求值(p,  x):
		return cn_array(numpy.polyval(p = p,  x = x))

	def 每月归还本金(利率,  每一个,  期数,  现在价值,  未来价值=0,  时间点='end'):
		return cn_array(numpy.ppmt(rate = 利率,  per = 每一个,  nper = 期数,  pv = 现在价值,  fv = 未来价值,  when = 时间点))

	def 打印选项(*args,  **kwargs):
		return cn_array(numpy.printoptions(*args,  **kwargs))

	def 乘积(序列,  轴=None,  数据类型=None,  输出=None,  保留维度集=numpy._NoValue,  初始值=numpy._NoValue,  条件=numpy._NoValue):
		return cn_array(numpy.prod(a = 序列,  axis = 轴,  dtype = 数据类型,  out = 输出,  keepdims = 保留维度集,  initial = 初始值,  where = 条件))

	def 乘法(*args,  **kwargs):
		return cn_array(numpy.product(*args,  **kwargs))

	def 极值差(序列,  轴=None,  输出=None,  保留维度集=numpy._NoValue):
		return cn_array(numpy.ptp(a = 序列,  axis = 轴,  out = 输出,  keepdims = 保留维度集))

	def 存值(序列,  切片,  v,  模式='raise'):
		return cn_array(numpy.put(a = 序列,  ind = 切片,  v = v,  mode = 模式))

	def 沿轴存值(序列,  切片集,  值集,  轴):
		return cn_array(numpy.put_along_axis(arr = 序列,  indices = 切片集,  values = 值集,  axis = 轴))

	def 当前价值(利率,  期数,  支付值,  未来价值=0,  时间点='end'):
		return cn_array(numpy.pv(rate = 利率,  nper = 期数,  pmt = 支付值,  fv = 未来价值,  when = 时间点))

	def 分位数(序列,  q,  轴=None,  输出=None,  输入超限=False,  插值='linear',  保留维度集=False):
		return cn_array(numpy.quantile(a = 序列,  q = q,  axis = 轴,  out = 输出,  overwrite_input = 输入超限,  interpolation = 插值,  keepdims = 保留维度集))

	def 利率(期数,  支付值,  现在价值,  未来价值,  时间点='end',  猜测=None,  公差=None,  maxiter=100):
		return cn_array(numpy.rate(nper = 期数,  pmt = 支付值,  pv = 现在价值,  fv = 未来价值,  when = 时间点,  guess = 猜测,  tol = 公差,  maxiter = maxiter))

	def 展开(序列,  顺序='C'):
		return cn_array(numpy.ravel(a = 序列,  order = 顺序))

	def 实部(值):
		return cn_array(numpy.real(val = 值))

	def 是否实部近似(序列,  公差=100):
		return cn_array(numpy.real_if_close(a = 序列,  tol = 公差))

	def 表序列(形状,  数据类型=None,  布夫=None,  偏移集=0,  步伐=None,  格式集=None,  名字集=None,  标题集=None,  字节顺序=None,  是否对齐=False,  顺序='C'):
		return cn_array(numpy.recarray(shape = 形状,  dtype = 数据类型,  buf = 布夫,  offset = 偏移集,  strides = 步伐,  formats = 格式集,  names = 名字集,  titles = 标题集,  byteorder = 字节顺序,  aligned = 是否对齐,  order = 顺序))

	def csv生成表(文件名,  **kwargs):
		return cn_array(numpy.recfromcsv(fname = 文件名,  **kwargs))

	def 文本生成表(文件名,  **kwargs):
		return cn_array(numpy.recfromtxt(fname = 文件名,  **kwargs))

	def 重复(序列,  重复,  轴=None):
		return cn_array(numpy.repeat(a = 序列,  repeats = 重复,  axis = 轴))

	def 引用(序列,  数据类型=None,  需求=None):
		return cn_array(numpy.require(a = 序列,  dtype = 数据类型,  requirements = 需求))

	def 变换形状(序列,  新形状,  顺序='C'):
		return cn_array(numpy.reshape(a = 序列,  newshape = 新形状,  order = 顺序))

	def 变换大小(序列,  新形状):
		return cn_array(numpy.resize(a = 序列,  new_shape = 新形状))

	def 滚动(序列,  移动,  轴=None):
		return cn_array(numpy.roll(a = 序列,  shift = 移动,  axis = 轴))

	def 滚动轴(序列,  轴,  开始=0):
		return cn_array(numpy.rollaxis(a = 序列,  axis = 轴,  start = 开始))

	def 根(p):
		return cn_array(numpy.roots(p = p))

	def 旋转90度(m,  k=1,  轴数=(0, 1)):
		return cn_array(numpy.rot90(m = m,  k = k,  axes = 轴数))

	def 四舍五入(序列,  小数点=0,  输出=None):
		return cn_array(numpy.round(a = 序列,  decimals = 小数点,  out = 输出))

	def 四舍五入_(序列,  小数点=0,  输出=None):
		return cn_array(numpy.round_(a = 序列,  decimals = 小数点,  out = 输出))

	def 行堆叠(元组):
		return cn_array(numpy.row_stack(tup = 元组))

	def 安全执行(来源):
		return cn_array(numpy.safe_eval(source = 来源))

	def 保存(文件,  序列,  是否可串行化=True,  固定导入=True):
		return cn_array(numpy.save(file = 文件,  arr = 序列,  allow_pickle = 是否可串行化,  fix_imports = 固定导入))

	def 保存文本(文件名,  X,  样式='%.18e',  定界符=' ',  新行='\n',  头部='',  尾部='',  注释='# ',  编码方式=None):
		return cn_array(numpy.savetxt(fname = 文件名,  X = X,  fmt = 样式,  delimiter = 定界符,  newline = 新行,  header = 头部,  footer = 尾部,  comments = 注释,  encoding = 编码方式))

	def 压缩保存(文件,  *args,  **kwds):
		return cn_array(numpy.savez(file = 文件,  *args,  **kwds))

	def 多序列压缩保存(文件,  *args,  **kwds):
		return cn_array(numpy.savez_compressed(file = 文件,  *args,  **kwds))

	def 标量类型转字符(变量类型):
		return cn_array(numpy.sctype2char(sctype = 变量类型))

	def 排序查询(序列,  v,  边='left',  排序器=None):
		return cn_array(numpy.searchsorted(a = 序列,  v = v,  side = 边,  sorter = 排序器))

	def 选择(条件列表,  选择列表,  默认=0):
		return cn_array(numpy.select(condlist = 条件列表,  choicelist = 选择列表,  default = 默认))

	def 设置打印选项(精度=None,  阈值=None,  边缘条目=None,  行宽=None,  抑制=None,  空值字符串=None,  信息字符串=None,  formatter=None,  符号=None,  floatmode=None,  **kwarg):
		return cn_array(numpy.set_printoptions(precision = 精度,  threshold = 阈值,  edgeitems = 边缘条目,  linewidth = 行宽,  suppress = 抑制,  nanstr = 空值字符串,  infstr = 信息字符串,  formatter = formatter,  sign = 符号,  floatmode = floatmode,  **kwarg))

	def 设置字符串函数(f,  代表=True):
		return cn_array(numpy.set_string_function(f = f,  repr = 代表))

	def 收进尺寸(尺寸):
		return cn_array(numpy.setbufsize(size = 尺寸))

	def 一维差集(序列1,  序列2,  默认唯一值=False):
		return cn_array(numpy.setdiff1d(ar1 = 序列1,  ar2 = 序列2,  assume_unique = 默认唯一值))

	def 设置错误(所有=None,  划分=None,  过度=None,  下=None,  无效=None):
		return cn_array(numpy.seterr(all = 所有,  divide = 划分,  over = 过度,  under = 下,  invalid = 无效))

	def 设置错误调用(功能):
		return cn_array(numpy.seterrcall(func = 功能))

	def 一维对称差集(序列1,  序列2,  默认唯一值=False):
		return cn_array(numpy.setxor1d(ar1 = 序列1,  ar2 = 序列2,  assume_unique = 默认唯一值))

	def 形状(序列):
		return cn_array(numpy.shape(a = 序列))

	def 显示配置():
		return cn_array(numpy.show_config())

	def 有符号整数():
		return cn_array(numpy.signedinteger())

	def 正弦除x(x):
		return cn_array(numpy.sinc(x = x))

	def 大小(序列,  轴=None):
		return cn_array(numpy.size(a = 序列,  axis = 轴))

	def 部分真(*args,  **kwargs):
		return cn_array(numpy.sometrue(*args,  **kwargs))

	def 排序(序列,  轴=-1,  种类=None,  顺序=None):
		return cn_array(numpy.sort(a = 序列,  axis = 轴,  kind = 种类,  order = 顺序))

	def 复数排序(序列):
		return cn_array(numpy.sort_complex(a = 序列))

	def 来源(对象,  输出=None):
		return cn_array(numpy.source(object = 对象,  output = 输出))

	def 拆分(序列,  切片集或分块集,  轴=0):
		return cn_array(numpy.split(ary = 序列,  indices_or_sections = 切片集或分块集,  axis = 轴))

	def 挤压维度(序列,  轴=None):
		return cn_array(numpy.squeeze(a = 序列,  axis = 轴))

	def 堆叠(序列集,  轴=0,  输出=None):
		return cn_array(numpy.stack(arrays = 序列集,  axis = 轴,  out = 输出))

	def 标准差(序列,  轴=None,  数据类型=None,  输出=None,  自由度=0,  保留维度集=numpy._NoValue):
		return cn_array(numpy.std(a = 序列,  axis = 轴,  dtype = 数据类型,  out = 输出,  ddof = 自由度,  keepdims = 保留维度集))

	def 求和(序列,  轴=None,  数据类型=None,  输出=None,  保留维度集=numpy._NoValue,  初始值=numpy._NoValue,  条件=numpy._NoValue):
		return cn_array(numpy.sum(a = 序列,  axis = 轴,  dtype = 数据类型,  out = 输出,  keepdims = 保留维度集,  initial = 初始值,  where = 条件))

	def 交换轴(序列,  轴1,  轴2):
		return cn_array(numpy.swapaxes(a = 序列,  axis1 = 轴1,  axis2 = 轴2))

	def 取值(序列,  切片集,  轴=None,  输出=None,  模式='raise'):
		return cn_array(numpy.take(a = 序列,  indices = 切片集,  axis = 轴,  out = 输出,  mode = 模式))

	def 沿轴移动(序列,  切片集,  轴):
		return cn_array(numpy.take_along_axis(arr = 序列,  indices = 切片集,  axis = 轴))

	def 张量点积(序列,  b,  轴数=2):
		return cn_array(numpy.tensordot(a = 序列,  b = b,  axes = 轴数))

	#ef test(label='fast',  verbose=1,  extra_argv=None,  doctests=False,  coverage=False,  durations=-1,  tests=None):
	#return cn_array(numpy.test(label = label,  verbose = verbose,  extra_argv = extra_argv,  doctests = doctests,  coverage = coverage,  durations = durations,  tests = tests))

	def 平铺(A,  重复):
		return cn_array(numpy.tile(A = A,  reps = 重复))

	def 对角线求和(序列,  偏移集=0,  轴1=0,  轴2=1,  数据类型=None,  输出=None):
		return cn_array(numpy.trace(a = 序列,  offset = 偏移集,  axis1 = 轴1,  axis2 = 轴2,  dtype = 数据类型,  out = 输出))

	def 转置(序列,  轴数=None):
		return cn_array(numpy.transpose(a = 序列,  axes = 轴数))

	def 梯度积分(y,  x=None,  dx=1.0,  轴=-1):
		return cn_array(numpy.trapz(y = y,  x = x,  dx = dx,  axis = 轴))

	def 三角序列(N,  M=None,  k=0,  数据类型=numpy.float):
		return cn_array(numpy.tri(N = N,  M = M,  k = k,  dtype = 数据类型))

	def 下三角序列(m,  k=0):
		return cn_array(numpy.tril(m = m,  k = k))

	def 下三角序列索引(n,  k=0,  m=None):
		return cn_array(numpy.tril_indices(n = n,  k = k,  m = m))

	def 下三角序列生成索引(序列,  k=0):
		return cn_array(numpy.tril_indices_from(arr = 序列,  k = k))

	def 修边去零(过滤器,  修边='fb'):
		return cn_array(numpy.trim_zeros(filt = 过滤器,  trim = 修边))

	def 上三角序列(m,  k=0):
		return cn_array(numpy.triu(m = m,  k = k))

	def 上三角序列索引(n,  k=0,  m=None):
		return cn_array(numpy.triu_indices(n = n,  k = k,  m = m))

	def 上三角序列生成索引(序列,  k=0):
		return cn_array(numpy.triu_indices_from(arr = 序列,  k = k))

	def 类别名(字符):
		return cn_array(numpy.typename(char = 字符))

	def 通用函数():
		return cn_array(numpy.ufunc())

	def 一维并集(序列1,  序列2):
		return cn_array(numpy.union1d(ar1 = 序列1,  ar2 = 序列2))

	def 去重(序列,  返回索引=False,  返回反转=False,  返回计数=False,  轴=None):
		return cn_array(numpy.unique(ar = 序列,  return_index = 返回索引,  return_inverse = 返回反转,  return_counts = 返回计数,  axis = 轴))

	def 无符号整数():
		return cn_array(numpy.unsignedinteger())

	def 角度定位(p,  派值=3.141592653589793,  轴=-1):
		return cn_array(numpy.unwrap(p = p,  discont = 派值,  axis = 轴))

	def 范德蒙矩阵(x,  N=None,  增加=False):
		return cn_array(numpy.vander(x = x,  N = N,  increasing = 增加))

	def 方差(序列,  轴=None,  数据类型=None,  输出=None,  自由度=0,  保留维度集=numpy._NoValue):
		return cn_array(numpy.var(a = 序列,  axis = 轴,  dtype = 数据类型,  out = 输出,  ddof = 自由度,  keepdims = 保留维度集))

	def 矢量化(py函数,  输出类型集=None,  文本=None,  排除=None,  缓存=False,  签名=None):
		return cn_array(numpy.vectorize(pyfunc = py函数,  otypes = 输出类型集,  doc = 文本,  excluded = 排除,  cache = 缓存,  signature = 签名))

	def 垂直拆分(序列,  切片集或分块集):
		return cn_array(numpy.vsplit(ary = 序列,  indices_or_sections = 切片集或分块集))

	def 垂直堆叠(元组):
		return cn_array(numpy.vstack(tup = 元组))

	def 谁(词典变量=None):
		return cn_array(numpy.who(vardict = 词典变量))

	def 同形零序列(序列,  数据类型=None,  顺序='K',  包含子类=True,  形状=None):
		return cn_array(numpy.zeros_like(a = 序列,  dtype = 数据类型,  order = 顺序,  subok = 包含子类,  shape = 形状))


	def 有序序列(*args, **kwargs):
		return cn_array(numpy.arange(*args, **kwargs))

	def 序列(对象,  数据类型=None,  复制=True,  顺序='K',  包含子类=False,  维度极小值=0):
		return cn_array(numpy.array(object = 对象,  dtype = 数据类型,  copy = 复制,  order = 顺序,  subok = 包含子类,  ndmin = 维度极小值))

	def 零序列(形状,  数据类型=numpy.float,  顺序='C'):
		return cn_array(numpy.zeros(shape = 形状,  dtype = 数据类型,  order = 顺序))

	def 空序列(形状,  数据类型=numpy.float,  顺序='C'):
		return cn_array(numpy.empty(shape = 形状,  dtype = 数据类型,  order = 顺序))

	#ef dtype(对象,  align=False,  复制=False):
	#return cn_array(numpy.dtype(obj = 对象,  align = align,  copy = 复制))

	def 字符串生成(字符串,  数据类型=numpy.float,  计数=-1,  分割=''):
		return cn_array(numpy.fromstring(string = 字符串,  dtype = 数据类型,  count = 计数,  sep = 分割))

	def 文件生成序列(文件,  数据类型=numpy.float,  计数=-1,  分割='',  偏移集=0):
		return cn_array(numpy.fromfile(file = 文件,  dtype = 数据类型,  count = 计数,  sep = 分割,  offset = 偏移集))

	def 缓冲生成序列(缓冲,  数据类型=numpy.float,  计数=-1,  偏移集=0):
		return cn_array(numpy.frombuffer(buffer = 缓冲,  dtype = 数据类型,  count = 计数,  offset = 偏移集))

	def 条件(条件,  x,  y):
		return cn_array(numpy.where(condition = 条件,  x = x,  y = y))

	def 复制到(目标,  src,  投射='same_kind',  条件=True):
		return cn_array(numpy.copyto(dst = 目标,  src = src,  casting = 投射,  where = 条件))

	def 拼接(序列,  轴=0,  输出=None):
		return cn_array(numpy.concatenate(array = 序列,  axis = 轴,  out = 输出))

	def 快速复制和转置(序列):
		return cn_array(numpy.fastCopyAndTranspose(a = 序列))

	def 行索引排序(键集,  轴=-1):
		return cn_array(numpy.lexsort(keys = 键集,  axis = 轴))

	def 设置数值算符(功能):
		return cn_array(numpy.set_numeric_ops(func = 功能))

	def 能否转换(从_,  到,  投射='safe'):
		return cn_array(numpy.can_cast(from_ = 从_,  to = 到,  casting = 投射))

	def 提升类型(type1,  type2):
		return cn_array(numpy.promote_types(type1 = type1,  type2 = type2))

	def 最小标量类型(序列):
		return cn_array(numpy.min_scalar_type(a = 序列))

	def 返回值数据类型(*arrays_and_dtypes):
		return cn_array(numpy.result_type(*arrays_and_dtypes))

	def 同形空序列(原型,  数据类型=None,  顺序='K',  包含子类=True,  形状=None):
		return cn_array(numpy.empty_like(prototype = 原型,  dtype = 数据类型,  order = 顺序,  subok = 包含子类,  shape = 形状))

	def 内积(序列,  b):
		return cn_array(numpy.inner(a = 序列,  b = b))

	def 点积(序列,  b,  输出=None):
		return cn_array(numpy.dot(a = 序列,  b = b,  out = 输出))

	def 向量点积(序列,  b):
		return cn_array(numpy.vdot(a = 序列,  b = b))

	def 迭代生成(可重复,  数据类型,  计数=-1):
		return cn_array(numpy.fromiter(iterable = 可重复,  dtype = 数据类型,  count = 计数))

	def 比较字符序列(序列,  b,  cmp_op,  rstrip):
		return cn_array(numpy.compare_chararrays(a = 序列,  b = b,  cmp_op = cmp_op,  rstrip = rstrip))

	def 遮罩存值(序列,  遮罩,  值集):
		return cn_array(numpy.putmask(a = 序列,  mask = 遮罩,  values = 值集))

	def 位反(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.bitwise_not(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 矩阵求积(x1,  x2,  输出=None,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.matmul(x1 = x1,  x2 = x2,  out = 输出,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 存储共享(序列,  b,  最大工作=None):
		return cn_array(numpy.shares_memory(a = 序列,  b = b,  max_work = 最大工作))

	def 是否存储共享(序列,  b,  最大工作=None):
		return cn_array(numpy.may_share_memory(a = 序列,  b = b,  max_work = 最大工作))

	def _添加新ufunc(ufunc,  new_docstring):
		return cn_array(numpy._add_newdoc_ufunc(ufunc = ufunc,  new_docstring = new_docstring))

	def 绝对值(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.absolute(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 相加(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.add(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 反余弦(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.arccos(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 反双曲余弦(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.arccosh(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 反正弦(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.arcsin(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 反双曲正弦(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.arcsinh(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 反正切(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.arctan(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 反正切2(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.arctan2(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 反双曲正切(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.arctanh(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 位与(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.bitwise_and(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 位或(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.bitwise_or(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 位异或(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.bitwise_xor(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 立方根(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.cbrt(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 上取整(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.ceil(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 共轭(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.conj(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 共轭复数(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.conjugate(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 复制符号(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.copysign(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 余弦(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.cos(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 双曲余弦(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.cosh(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 角度转弧度(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.deg2rad(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 角度数(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.degrees(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 求商(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.divide(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 余数除法(x1,  x2,  输出=(None, None),  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.divmod(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 相等(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.equal(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 自然指数(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.exp(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 底2指数(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.exp2(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 计算(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.expm1(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 浮点绝对值(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.fabs(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 下取整(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.floor(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 下取整商(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.floor_divide(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 浮点指数(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.float_power(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 最大值序列(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.fmax(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 最小值序列(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.fmin(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 余数序列(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.fmod(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 生成底2指数(x,  输出=(None, None),  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.frexp(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def py函数生成(功能,  输入数,  输出数):
		return cn_array(numpy.frompyfunc(func = 功能,  nin = 输入数,  nout = 输出数))

	def 最大公约数(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.gcd(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 获取错误对象():
		return cn_array(numpy.geterrobj())

	def 大于(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.greater(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 大于等于(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.greater_equal(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 重瓣(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.heaviside(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 斜边(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.hypot(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 位否(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.invert(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 是否有穷(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.isfinite(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 是否无穷(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.isinf(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 是否空值(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.isnan(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 是否时间空值(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.isnat(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 索引多键排序(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.lcm(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 加载底2指数(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.ldexp(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 左移(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.left_shift(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 小于(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.less(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 小于等于(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.less_equal(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 自然对数(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.log(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 常用对数(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.log10(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 加一对数(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.log1p(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 底2对数(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.log2(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 底2对数加法器(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.logaddexp(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 自然对数加法器(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.logaddexp2(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 逻辑与(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.logical_and(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 逻辑非(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.logical_not(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 逻辑或(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.logical_or(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 逻辑异或(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.logical_xor(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 最大值(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.maximum(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 最小值(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.minimum(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 求余(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.mod(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 浮点余数(x,  输出=(None, None),  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.modf(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 求积(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.multiply(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 负数(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.negative(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 浮点等差数(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.nextafter(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 不相等(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.not_equal(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 正值(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.positive(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 指数(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.power(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 弧度转角度(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.rad2deg(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 弧度(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.radians(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 倒数(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.reciprocal(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 余数(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.remainder(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 右移(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.right_shift(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 取整(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.rint(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 设置错误对象(错误对象):
		return cn_array(numpy.seterrobj(errobj = 错误对象))

	def 符号(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.sign(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 符号位(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.signbit(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 正弦(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.sin(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 双曲正弦(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.sinh(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 间距(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.spacing(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 平方根(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.sqrt(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 平方(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.square(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 求差(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.subtract(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 正切(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.tan(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 双曲正切(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.tanh(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 真除法(x1,  x2,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.true_divide(x1 = x1,  x2 = x2,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 截断(x,  输出=None,  条件=True,  投射='same_kind',  顺序='K',  数据类型=None,  包含子类=True):
		return cn_array(numpy.trunc(x = x,  out = 输出,  where = 条件,  casting = 投射,  order = 顺序,  dtype = 数据类型,  subok = 包含子类))

	def 日期时间数据(数据类型):
		return cn_array(numpy.datetime_data(dtype = 数据类型))

	def 日期时间当字符串(序列,  单元=None,  时区='naive',  投射='same_kind'):
		return cn_array(numpy.datetime_as_string(arr = 序列,  unit = 单元,  timezone = 时区,  casting = 投射))

	def 工作日推算(日期,  偏移集,  滚动='raise',  星期掩模='1111100',  假期集=None,  工作日=None,  输出=None):
		return cn_array(numpy.busday_offset(dates = 日期,  offsets = 偏移集,  roll = 滚动,  weekmask = 星期掩模,  holidays = 假期集,  busdaycal = 工作日,  out = 输出))

	def 工作日计数(开始日期,  结束日期,  星期掩模='1111100',  假期集=[],  工作日=None,  输出=None):
		return cn_array(numpy.busday_count(begindates = 开始日期,  enddates = 结束日期,  weekmask = 星期掩模,  holidays = 假期集,  busdaycal = 工作日,  out = 输出))

	def 是否工作日(日期,  星期掩模='1111100',  假期集=None,  工作日=None,  输出=None):
		return cn_array(numpy.is_busday(dates = 日期,  weekmask = 星期掩模,  holidays = 假期集,  busdaycal = 工作日,  out = 输出))

	def busdaycalendar(星期掩模='1111100',  假期集=None):
		return cn_array(numpy.busdaycalendar(weekmask = 星期掩模,  holidays = 假期集))

	def 一维索引(多重索引,  维度集,  模式='raise',  顺序='C'):
		return cn_array(numpy.ravel_multi_index(multi_index = 多重索引,  dims = 维度集,  mode = 模式,  order = 顺序))

	def 多维索引(切片集,  形状,  顺序='C'):
		return cn_array(numpy.unravel_index(indices = 切片集,  shape = 形状,  order = 顺序))

	def 块计数(x,  权重=None,  最小长度=0):
		return cn_array(numpy.bincount(x = x,  weights = 权重,  minlength = 最小长度))

	def 添加帮助(对象,  docstring):
		return cn_array(numpy.add_docstring(obj = 对象,  docstring = docstring))

	def 添加新ufunc(ufunc,  new_docstring):
		return cn_array(numpy.add_newdoc_ufunc(ufunc = ufunc,  new_docstring = new_docstring))

	def 打包位(序列,  轴=None,  位序='big'):
		return cn_array(numpy.packbits(a = 序列,  axis = 轴,  bitorder = 位序))

	def 拆包位(序列,  轴=None,  计数=None,  位序='big'):
		return cn_array(numpy.unpackbits(a = 序列,  axis = 轴,  count = 计数,  bitorder = 位序))
	