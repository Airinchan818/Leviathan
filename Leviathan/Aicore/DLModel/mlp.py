from Leviathan import littlelearn as ll 
import jax.numpy as jnp 
from typing import Literal
class ModelBase (ll.DeepLearning.layers.Component) :
    "MLP Base Model"
    def __init__(self) :
        super().__init__()

        self.list_of_parameter = []


    def return_Tensor_jaxArr(self) :
        param = self.parameter()
        jArr = [] 
        for p in param :
            jArr.append(jnp.array(p.tensor)) 

        return jArr 


class LinearModel (ModelBase) :
    "Linear Model for simple data execution"
    def __init__(self,data_dim) :
        super().__init__()
        self.name = self.__init__.__name__
        self.w = ll.DeepLearning.layers.Parameter(
            tensor=ll.randn(
                shape=(data_dim,1)
            )
        )
        self.b = ll.DeepLearning.layers.Parameter(
            tensor = ll.randn(
                shape=(data_dim,1)
            )
        )

    def forwardpass(self,x) :
        return ll.matmul(x,self.w) + self.b 

class LineRelu (ll.DeepLearning.layers.Component) :
    def __init__(self,in_dim,out_dim,add_bias=True) :
        super().__init__()
        self.layer = ll.DeepLearning.layers.Linear(in_dim,out_dim,add_bias=add_bias)

    def forwardpass(self,x):
        x = self.layer(x) 
        return ll.DeepLearning.activations.relu(x)
    

class ClassicMLP(ModelBase) : 
    "Basic MLP Model for low to mid task "
    def __init__(self,data_dim,depth_model,factor_expand_dim : int,bias : bool = True,
                 variant : Literal['mc','bin','reg','sig']='reg') :
        super().__init__()
        self.factor_expanddim = factor_expand_dim
        self.__first_layer_dim = 2 * data_dim
        self.bias = bias 
        self.layer1 = ll.DeepLearning.layers.Linear(
            input_dim=data_dim, output_dim=self.__first_layer_dim,add_bias=bias
        )
        mid_layer =[]
        out_dim = 32 
        for _ in range(depth_model) :
            
            mid_layer.append(LineRelu(self.__first_layer_dim,out_dim,add_bias=bias))
            out_dim *= factor_expand_dim 

        self.mid_layer = ll.DeepLearning.layers.Sequential(mid_layer)

        self.last_layer = ll.DeepLearning.layers.Linear(
            input_dim=out_dim, output_dim=1,add_bias=bias
        )
        self.variant_out = variant

    def forwardpass(self,x) :
        x = self.layer1(x)
        x = self.mid_layer(x)
        x = self.last_layer(x)
        if self.variant_out == 'mc' :
            softmax_out = ll.DeepLearning.activations.softmax(x)
            return {
                "softmax_out" : softmax_out,
                "logits" : x 
            } 
        elif self.variant_out == 'bin' :
            sigmoid_out = ll.DeepLearning.activations.sigmoid(x)
            return {
                "sigmoid_out" : sigmoid_out,
                "logits" : x
            } 
        
        elif self.variant_out == 'sig' :
            tanh_out = ll.DeepLearning.activations.tanh(x)
            return {
                "tanh_out" : tanh_out,
                "logits" : x
            }

        else :
            return x 





class MiniMLPBlock (ll.DeepLearning.layers.Component) :
    def __init__ (self,input_dim,output_dim,add_bias = False) :
        super().__init__()
        self.layer1 = ll.DeepLearning.layers.Linear(input_dim,16,add_bias=add_bias)
        self.layer2 = ll.DeepLearning.layers.Linear(16,32,add_bias=add_bias)
        self.layer3 = ll.DeepLearning.layers.Linear(32,output_dim,add_bias=add_bias)

    def forwardpass(self,x) :
        x = self.layer1(x)
        x = ll.DeepLearning.activations.relu(x)
        x = self.layer2(x)
        x = ll.DeepLearning.activations.relu(x)
        x = self.layer3(x)
        return x

class ReRandom(ll.DeepLearning.layers.Component) :
    def __init__(self,input_dim,init_ = 1e-3,scale_to : Literal ['main','factor'] = 'factor' ) :
        super().__init__()
        self.factors = ll.DeepLearning.layers.Parameter(
            tensor=ll.normal(mean=0,std=init_,shape=(input_dim,))
        ) 
        self.__comand = scale_to

    def forwardpass(self,x,y) : 
        if self.__comand == 'factor' : 
            return x + (y * self.factors)

        else :
            return (x * self.factors) + y   

class ReZeros (ll.DeepLearning.layers.Component) :

    def __init__(self,input_dim,scale_to : Literal ['main','factor'] = 'factor' ) :
        super().__init__()
        self.factors = ll.DeepLearning.layers.Parameter(
            tensor=ll.zeros(shape=(input_dim))
        ) 
        self.__comand = scale_to

    def forwardpass(self,x,y) :
        if self.__comand == 'factor' :
            return x + (y * self.factors)
        else :
            return (x * self.factors) + y 
        

class MultiBlockMLP (ModelBase) :
    def __init__(self,data_dim,out_dim,block_num,block_out = 16,bias = False,
                 variant : Literal['mc','bin','reg','sig']='reg',
                 norm_variant : Literal['scale','norm','rms'] = 'scale',
                  res_variant : Literal['general','rerandom','zeros','scale'] = 'general'
                   ,res_factor : Literal ['main','factor'] = 'factor'   ) :
        super().__init__()
        self.head_first = ll.DeepLearning.layers.Linear(data_dim,64)
        if res_variant == 'general' :
            self.resmodule = None

        elif res_variant == 'rerandom' :
            self.resmodule = ReRandom(block_out,scale_to=res_factor) 

         
         
               
  

 

