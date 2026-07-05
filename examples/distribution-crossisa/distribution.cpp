import std;
using namespace std;
namespace rs=std::ranges;
using Real=double;
using Reals=vector<Real>;
using Integer=long long int;
using YMultiplicityFn=map<Real,tuple<Integer,Real>>;
using DC=piecewise_constant_distribution<Real>;
using DD=discrete_distribution<Integer>;
using DL=piecewise_linear_distribution<Real>;
namespace {
	Real dlPDF(const Real& x, const DL& d)
	{
		const auto& b{d.intervals()}; const auto& r{d.densities()};
		if(b.empty()) throw invalid_argument("dlPDF: empty");
		if(x<b.front()||x>b.back()) return 0.0;
		const auto i{distance(b.cbegin(),rs::lower_bound(b,x))};
		return x==b[i]?r[i]:(r[i-1]*(b[i]-x)+r[i]*(x-b[i-1]))/(b[i]-b[i-1]);
	}
	Real dlDF(const Real& x, const DL& d)
	{
		const auto& b{d.intervals()}; const auto& r{d.densities()};
		if(b.empty()) throw invalid_argument("dlDF: empty");
		if(x<=b.front()) return 0.0; if(x>=b.back()) return 1.0;
		const auto i{distance(b.begin(),rs::lower_bound(b,x))};
		const auto df{accumulate(b.cbegin(), b.cbegin()+i-1,0.0,[&]
		(const auto& acc, const auto& v){const auto k{distance(&b[0],&v)};
		return acc+0.5*(r[k]+r[k+1])*(*(&v+1)-v);})};
		return df+((x-b[i-1])/(b[i]-b[i-1]))*(r[i-1]*b[i]
		-r[i]*b[i-1]+0.5*(x+b[i-1])*(r[i]-r[i-1]));
	}
	Real dcPDF(const Real& x, const DC& d)
	{
		const auto& is{d.intervals()};
		if(is.empty()) throw invalid_argument("dcPDF: empty");
		if(x<is.front()||x>=is.back()) return 0.0;
		const auto i{distance(is.cbegin(),rs::lower_bound(is,x))};
		return i==is.size()-1?d.densities().back():d.densities()[i];
	}
	Real dcDF(const Real& x, const DC& d)
	{
		const auto& b{d.intervals()};
		if(b.empty()) throw invalid_argument("dcDF: empty");
		if(x<=b.front()) return 0.0; if(x>=b.back()) return 1.0;
		const auto i{distance(b.begin(),rs::lower_bound(b,x))};
		const auto df{accumulate(b.cbegin(),b.cbegin()+i-1,0.0,[&]
		(const auto& acc, const auto& v){const auto k{distance(&b[0],&v)};
		return acc+d.densities()[k]*(*(&v+1)-v);})};
		return df+(x-b[i-1])*d.densities()[i-1];
	}
	Real ddPMF(const Real& x, const DD& d, const Reals& b)
	{
		const auto& p{d.probabilities()};
		if(p.empty()) throw invalid_argument("ddPMF: empty");
		if(p.size()!=b.size()) throw invalid_argument("ddPMF: p and b "
		" have different sizes " + to_string(p.size()) + " and "
		+ to_string(b.size()));
		const auto& i{rs::lower_bound(b, x)};
		if(i==b.cend()||*i!=x) return 0.0; return p[distance(b.cbegin(),i)];
	}
	Real ddDF(const Real& x, const DD& d, const Reals& b)
	{
		const auto& p{d.probabilities()};
		if(p.empty()) throw invalid_argument("ddDF: empty");
		if(p.size()!=b.size()) throw invalid_argument("ddDF: p and b "
		" have different sizes " + to_string(p.size()) + " and "
		+ to_string(b.size()));
		const auto& i{rs::lower_bound(b, x)};
		if(i==b.cend()) return 1.0;
		const auto k{distance(b.cbegin(),i)};
		return accumulate(p.cbegin(),p.cbegin()+k+1,0.0);
	}
	Real ankFn(const Real& x, const YMultiplicityFn& ymf)
	{
		if(ymf.empty()) throw invalid_argument("ankFn: empty");
		if(x<(*ymf.cbegin()).first) return 0.0;
		const auto& i{ymf.lower_bound(x)};
		return i==ymf.cend()?1.0:get<1>((*i).second);
	}
}
int main(int argc, char *argv[])
{
	try {
		if(argc<2) {cout<<"Usage: "<<argv[0]<<" intervals"<<endl;return 0;}
		const auto INTERVALS{stoul(argv[1])};
		if(INTERVALS<1) throw invalid_argument("intervals must be positive");
		YMultiplicityFn ymf{}; Real value{};
		while(cin>>value) {get<0>(ymf[value])++;}
		Reals boundaries{}; Reals weights{};
		boundaries.reserve(ymf.size()); weights.reserve(boundaries.size());
		for(const auto& d:ymf) {
			boundaries.push_back(d.first);
			weights.push_back(static_cast<Real>(get<0>(d.second)));
		}
		const Real total{accumulate(weights.cbegin(), weights.cend(), 0.0)};
		Real Fn{};
		for(auto& d:ymf){Fn+=get<0>(d.second)/total;get<1>(d.second)=Fn;}
		Reals xs(boundaries.cbegin(), boundaries.cend());
		const auto dx{(boundaries.back()-boundaries.front())/INTERVALS};
		xs.reserve(INTERVALS+boundaries.size()-1);
		Real v{boundaries.front()+dx};
		for(Integer p{1};p<INTERVALS;++p,v+=dx) xs.push_back(v);
		rs::sort(xs);
		DC dc{boundaries.cbegin(), boundaries.cend(), weights.cbegin()};
		DD dd{weights.cbegin(), weights.cend()};
		DL dl{boundaries.cbegin(), boundaries.cend(), weights.cbegin()};
		for(const auto& x:xs) {
			constexpr int W{25}, P{16};
			cout<<setw(W)<<setprecision(P)<<x<<" "
			<<setw(W)<<setprecision(P)<<dlPDF(x,dl)<<" "
			<<setw(W)<<setprecision(P)<<dlDF(x,dl)<<" "
			<<setw(W)<<setprecision(P)<<dcPDF(x,dc)<<" "
			<<setw(W)<<setprecision(P)<<dcDF(x,dc)<<" "
			<<setw(W)<<setprecision(P)<<ddPMF(x,dd,boundaries)<<" "
			<<setw(W)<<setprecision(P)<<ddDF(x,dd,boundaries)<<" "
			<<setw(W)<<setprecision(P)<<ankFn(x,ymf)<<endl;
		}
	}
	catch(const exception& e) {cerr << e.what() << endl;return 1;}
	catch(...) {cerr << "Uknown exception" << endl;return 2;}
	return 0;
}
