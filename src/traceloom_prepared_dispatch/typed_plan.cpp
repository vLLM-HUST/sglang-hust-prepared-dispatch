#include <torch/extension.h>
#include <ATen/core/dispatch/Dispatcher.h>
namespace py=pybind11;
using V=std::optional<std::vector<c10::SymInt>>;
V seq(py::handle x){ if(x.is_none()) return std::nullopt; V r=std::vector<c10::SymInt>(); for(auto a:x.cast<py::sequence>())r->emplace_back(a.cast<int64_t>()); return r; }
at::OptionalSymIntArrayRef seq_ref(const V& v){if(!v)return std::nullopt;return at::OptionalSymIntArrayRef(*v);}
using Sig=std::tuple<at::Tensor,at::Tensor>(const at::Tensor&,const at::Tensor&,const at::Tensor&,const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,at::OptionalSymIntArrayRef,at::OptionalSymIntArrayRef,const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,at::OptionalSymIntArrayRef,const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,const std::optional<at::Tensor>&,int64_t,double,int64_t,int64_t,c10::string_view,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,const std::optional<at::Tensor>&,at::TensorList);
bool validate_schema(){auto op=c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_fused_infer_attention_score","out").typed<Sig>();return true;}
struct Plan {
at::Tensor query;
at::Tensor key;
at::Tensor value;
std::optional<at::Tensor> pse_shift;
std::optional<at::Tensor> atten_mask;
std::optional<std::vector<c10::SymInt>> actual_seq_lengths;
std::optional<std::vector<c10::SymInt>> actual_seq_lengths_kv;
std::optional<at::Tensor> dequant_scale1;
std::optional<at::Tensor> quant_scale1;
std::optional<at::Tensor> dequant_scale2;
std::optional<at::Tensor> quant_scale2;
std::optional<at::Tensor> quant_offset2;
std::optional<at::Tensor> antiquant_scale;
std::optional<at::Tensor> antiquant_offset;
std::optional<at::Tensor> key_antiquant_scale;
std::optional<at::Tensor> key_antiquant_offset;
std::optional<at::Tensor> value_antiquant_scale;
std::optional<at::Tensor> value_antiquant_offset;
std::optional<at::Tensor> block_table;
std::optional<at::Tensor> query_padding_size;
std::optional<at::Tensor> kv_padding_size;
std::optional<at::Tensor> key_shared_prefix;
std::optional<at::Tensor> value_shared_prefix;
std::optional<std::vector<c10::SymInt>> actual_shared_prefix_len;
std::optional<at::Tensor> query_rope;
std::optional<at::Tensor> key_rope;
std::optional<at::Tensor> key_rope_antiquant_scale;
int64_t num_heads;
double scale;
int64_t pre_tokens;
int64_t next_tokens;
std::string input_layout;
int64_t num_key_value_heads;
int64_t sparse_mode;
int64_t inner_precise;
int64_t block_size;
int64_t antiquant_mode;
int64_t key_antiquant_mode;
int64_t value_antiquant_mode;
bool softmax_lse_flag;
std::optional<at::Tensor> workspace;
std::vector<at::Tensor> out;
bool guard; std::vector<int64_t> last;
Plan(py::dict d,bool g):guard(g){
query=d["query"].cast<at::Tensor>();
key=d["key"].cast<at::Tensor>();
value=d["value"].cast<at::Tensor>();
pse_shift=d["pse_shift"].cast<std::optional<at::Tensor>>();
atten_mask=d["atten_mask"].cast<std::optional<at::Tensor>>();
actual_seq_lengths=seq(d["actual_seq_lengths"]);
actual_seq_lengths_kv=seq(d["actual_seq_lengths_kv"]);
dequant_scale1=d["dequant_scale1"].cast<std::optional<at::Tensor>>();
quant_scale1=d["quant_scale1"].cast<std::optional<at::Tensor>>();
dequant_scale2=d["dequant_scale2"].cast<std::optional<at::Tensor>>();
quant_scale2=d["quant_scale2"].cast<std::optional<at::Tensor>>();
quant_offset2=d["quant_offset2"].cast<std::optional<at::Tensor>>();
antiquant_scale=d["antiquant_scale"].cast<std::optional<at::Tensor>>();
antiquant_offset=d["antiquant_offset"].cast<std::optional<at::Tensor>>();
key_antiquant_scale=d["key_antiquant_scale"].cast<std::optional<at::Tensor>>();
key_antiquant_offset=d["key_antiquant_offset"].cast<std::optional<at::Tensor>>();
value_antiquant_scale=d["value_antiquant_scale"].cast<std::optional<at::Tensor>>();
value_antiquant_offset=d["value_antiquant_offset"].cast<std::optional<at::Tensor>>();
block_table=d["block_table"].cast<std::optional<at::Tensor>>();
query_padding_size=d["query_padding_size"].cast<std::optional<at::Tensor>>();
kv_padding_size=d["kv_padding_size"].cast<std::optional<at::Tensor>>();
key_shared_prefix=d["key_shared_prefix"].cast<std::optional<at::Tensor>>();
value_shared_prefix=d["value_shared_prefix"].cast<std::optional<at::Tensor>>();
actual_shared_prefix_len=seq(d["actual_shared_prefix_len"]);
query_rope=d["query_rope"].cast<std::optional<at::Tensor>>();
key_rope=d["key_rope"].cast<std::optional<at::Tensor>>();
key_rope_antiquant_scale=d["key_rope_antiquant_scale"].cast<std::optional<at::Tensor>>();
num_heads=d["num_heads"].cast<int64_t>();
scale=d["scale"].cast<double>();
pre_tokens=d["pre_tokens"].cast<int64_t>();
next_tokens=d["next_tokens"].cast<int64_t>();
input_layout=d["input_layout"].cast<std::string>();
num_key_value_heads=d["num_key_value_heads"].cast<int64_t>();
sparse_mode=d["sparse_mode"].cast<int64_t>();
inner_precise=d["inner_precise"].cast<int64_t>();
block_size=d["block_size"].cast<int64_t>();
antiquant_mode=d["antiquant_mode"].cast<int64_t>();
key_antiquant_mode=d["key_antiquant_mode"].cast<int64_t>();
value_antiquant_mode=d["value_antiquant_mode"].cast<int64_t>();
softmax_lse_flag=d["softmax_lse_flag"].cast<bool>();
workspace=d["workspace"].cast<std::optional<at::Tensor>>();
out=d["out"].cast<std::vector<at::Tensor>>(); TORCH_CHECK(out.size()==2);
}
void run(py::args a,py::kwargs k){
 TORCH_CHECK(a.size()==3,"Expected 3 captured positional tensor arguments");
 auto qseq=k.contains("actual_seq_lengths") ? seq(k["actual_seq_lengths"]) : actual_seq_lengths;
 auto kvseq=k.contains("actual_seq_lengths_kv") ? seq(k["actual_seq_lengths_kv"]) : actual_seq_lengths_kv;
 if(guard){last.clear();if(kvseq)for(auto x:*kvseq)last.push_back(x.expect_int());}
 actual_seq_lengths=std::move(qseq);actual_seq_lengths_kv=std::move(kvseq);
 py::gil_scoped_release release;
 static auto op=c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_fused_infer_attention_score","out").typed<Sig>();
 op.call(query,
key,
value,
pse_shift,
atten_mask,
seq_ref(actual_seq_lengths),
seq_ref(actual_seq_lengths_kv),
dequant_scale1,
quant_scale1,
dequant_scale2,
quant_scale2,
quant_offset2,
antiquant_scale,
antiquant_offset,
key_antiquant_scale,
key_antiquant_offset,
value_antiquant_scale,
value_antiquant_offset,
block_table,
query_padding_size,
kv_padding_size,
key_shared_prefix,
value_shared_prefix,
seq_ref(actual_shared_prefix_len),
query_rope,
key_rope,
key_rope_antiquant_scale,
num_heads,
scale,
pre_tokens,
next_tokens,
input_layout,
num_key_value_heads,
sparse_mode,
inner_precise,
block_size,
antiquant_mode,
key_antiquant_mode,
value_antiquant_mode,
softmax_lse_flag,
workspace,
out);
}
};
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("validate_schema",&validate_schema);py::class_<Plan>(m,"Plan").def(py::init<py::dict,bool>()).def("__call__",&Plan::run).def("__getattr__",[](Plan&,const std::string& key){if(key=="__name__")return py::str("npu_fused_infer_attention_score.out");throw py::attribute_error(key);}).def_readonly("last_kv",&Plan::last);}
