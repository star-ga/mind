import std;
using namespace std;
namespace {
	// Command line arguments. ____________________________________________
	enum class P{OUT,DISTR,PS,SEED,ACC,MARGIN,TS,RATE,DAILY,IS,B,SZ};
	constexpr auto OUT=to_underlying(P::OUT),DISTR=to_underlying(
	P::DISTR),PS=to_underlying(P::PS),SEED=to_underlying(P::SEED),
	ACC=to_underlying(P::ACC),MARGIN=to_underlying(P::MARGIN),TS=
	to_underlying(P::TS),RATE=to_underlying(P::RATE),DAILY=
	to_underlying(P::DAILY),IS=to_underlying(P::IS),
	B=to_underlying(P::B),SZ=to_underlying(P::SZ);
	using A=tuple<char,char,size_t,int,double,double,size_t,double,size_t,
	size_t,double>;
	void help(ostream& os, int argc, char *argv[], const A& a)
	{
		os<<"Usage: "<<argv[0]<<" OUT DISTR PS SEED ACC MARGIN TS "
		"RATE DAILY B"
		<<"\n  OUT    - report [l]ine, [a]cc distr, [s]gr distr: "<<get<OUT>(a)
		<<"\n  DISTR  - [c]onstant, [d]iscrete, [l]inear: "<<get<DISTR>(a)
		<<"\n  PS     - number of paths > 0: "<<get<PS>(a)
		<<"\n  SEED   - seed > 0: "<<get<SEED>(a)
		<<"\n  ACC    - initial account > 0: "<<get<ACC>(a)
		<<"\n  MARGIN - initial margin > 0: "<<get<MARGIN>(a)
		<<"\n  TS     - number of trades > 0: "<<get<TS>(a)
		<<"\n  RATE   - borrowing interest rate >= 0: "<<get<RATE>(a)
		<<"\n  DAILY  - number of trades per day > 0: "<<get<DAILY>(a)
		<<"\n  IS     - (Xmax-Xmin)/IS, intervals > 0: "<<get<IS>(a)
		<<"\n  B      - fraction >= 0: "<<get<B>(a)<<endl;
	}
	void validate(const A& a)
	{
		if(get<OUT>(a)!='l'&&get<OUT>(a)!='a'&&get<OUT>(a)!='s')
		throw invalid_argument(string("err: OUT=")+get<OUT>(a));
		if(get<DISTR>(a)!='c'&&get<DISTR>(a)!='l'&&get<DISTR>(a)!='d')
		throw invalid_argument(string("err: DISTR=")+get<DISTR>(a));
		if(get<PS>(a)<1) throw
		invalid_argument("err: PS="+to_string(get<PS>(a)));
		if(get<SEED>(a)<1) throw
		invalid_argument("err: SEED="+to_string(get<SEED>(a)));
		if(get<ACC>(a)<=0.0) throw
		invalid_argument("err: ACC="+to_string(get<ACC>(a)));
		if(get<MARGIN>(a)<=0.0) throw
		invalid_argument("err: MARGIN="+to_string(get<MARGIN>(a)));
		if(get<TS>(a)<1) throw
		invalid_argument("err: TS="+to_string(get<TS>(a)));
		if(get<RATE>(a)<0.0) throw
		invalid_argument("err: RATE="+to_string(get<RATE>(a)));
		if(get<DAILY>(a)<1) throw
		invalid_argument("err: DAILY="+to_string(get<DAILY>(a)));
		if(get<IS>(a)<1) throw
		invalid_argument("err: IS="+to_string(get<IS>(a)));
		if(get<B>(a)<0.0) throw
		invalid_argument("err: B="+to_string(get<B>(a)));
	}
	// ICH framework. I - interface, C - concrete, H - handle. ____________
	class IDistr{
	public:
		IDistr()=default;
		IDistr(const IDistr&)=default;
		IDistr(IDistr&&)=default;
		IDistr& operator=(const IDistr&)=default;
		IDistr& operator=(IDistr&&)=default;
		virtual ~IDistr() noexcept=default;
		virtual double operator()(mt19937& e)=0;
	};
	template<typename Distr>
	class CDistr final:public IDistr{
	public:
		template<typename...Args>
		explicit CDistr(Args&&...args):d_(forward<Args>(args)...){}
		double operator()(mt19937& e)override{return d_(e);}
	private:
		Distr d_{};
	};
	using V=vector<double>;
	using SpecialDistr=discrete_distribution<size_t>;
	template<> class CDistr<SpecialDistr> final:public IDistr{
	public:
		CDistr():b_{0},d_{}{}
		CDistr(const V& b, const V& w):b_{b},d_{w.cbegin(),w.cend()}{}
		double operator()(mt19937& e)override{return b_[d_(e)];}
	private:
		V b_{}; SpecialDistr d_{};
	};
	class HDistr final{
	public:
		HDistr()=default;
		template<typename D,typename...Args>
		HDistr(type_identity<D>,Args&&...args):p_{make_unique<CDistr<
		type_identity_t<D>>>(forward<Args>(args)...)}{}
		double operator()(mt19937& e){if(!p_) throw invalid_argument(
		"HDistr is unitialized");return p_->operator()(e);}
	private:
		unique_ptr<IDistr> p_{};
	};
	using L=type_identity<CDistr<piecewise_linear_distribution<>>>;
	using C=type_identity<CDistr<piecewise_constant_distribution<>>>;
	using D=type_identity<CDistr<discrete_distribution<size_t>>>;
	// Iteral template for a function of one variable. ____________________
	template<typename T, typename F>
	T iteral(size_t numberOfIterations,const T& initialValue,F f)
	{
		T tmp{initialValue};
		for(size_t i{}; i<numberOfIterations; ++i) tmp=f(tmp);
		return tmp;
	}
	// Summation and Statistics. __________________________________________
	struct SumCompensation {double sum{};double compensation{};};
	SumCompensation sumKahan(SumCompensation sc,double value)
	{
		volatile double y{value-sc.compensation};
		volatile double t{sc.sum+y};
		sc.compensation=(t-sc.sum)-y; sc.sum=t;
		return sc;
	}
	using SampleDistribution=map<double,size_t>;
	size_t sampleSize(const SampleDistribution& d)
	{
		return accumulate(d.cbegin(),d.cend(),0uz,[](const auto& acc,const
		auto& vm){return acc+vm.second;});
	}
	double average(const SampleDistribution& d)
	{
		const auto sz{sampleSize(d)};
		const auto s{std::accumulate(d.cbegin(),d.cend(),SumCompensation{},
		[](const auto& acc, const auto& vm)
		{return sumKahan(acc, vm.first*vm.second);})};
		if(!sz)throw invalid_argument("average: empty distribution");
		return s.sum/sz;
	}
	double stdev(const SampleDistribution& d)
	{
		const auto sz{sampleSize(d)};
		if(!sz)throw invalid_argument("stdev: empty distribution");
		if(sz<2)return 0.0;
		const auto avg{average(d)};
		const auto s2{std::accumulate(d.cbegin(),d.cend(),SumCompensation{},
		[](const auto& acc, const auto& vm)
		{return sumKahan(acc, vm.first*vm.first*vm.second);})};
		const auto dev{(s2.sum/sz-avg*avg)/(sz-1)};
		return dev<0.0?0.0:sqrt(dev);
	}
	ostream& operator<<(ostream& os,const SampleDistribution& d)
	{
		if(os){
			constexpr int W{25},P{16};
			for(const auto& p:d)
			os<<setw(W)<<setprecision(P)<<p.first<<" "<<setw(W)<<p.second
			<<endl;
		}
		return os;
	}
	SampleDistribution toEqualIS(const SampleDistribution& d, size_t is)
	{
		if(is<1||d.size()<2)return d;
		const auto dv{((*d.crbegin()).first-(*d.cbegin()).first)/is};
		SampleDistribution tmp{};
		for(const auto& p:d) {
			const auto i{floor((p.first-(*d.cbegin()).first)/dv)};
			tmp[(*d.cbegin()).first+i*dv]+=p.second;
		}
		return tmp;
	}
}
int main(int argc, char *argv[])
{
	try {
		A a{'l','l',10000,1234567,100000.0,6600.0,10,0.1,100,1,1.0};
		if(argc<SZ+1){help(cout,argc,argv,a);return 0;}
		get<OUT>(a)=tolower(argv[OUT+1][0]);get<DISTR>(a)=tolower(
		argv[DISTR+1][0]);get<PS>(a)=stoul(argv[PS+1]);get<SEED>(a)=
		stoul(argv[SEED+1]);get<ACC>(a)=stod(argv[ACC+1]);get<MARGIN>(a)=
		stod(argv[MARGIN+1]);get<TS>(a)=stoul(argv[TS+1]);get<RATE>(a)=
		stod(argv[RATE+1]);get<DAILY>(a)=stoul(argv[DAILY+1]);get<IS>(a)=
		stoul(argv[IS+1]);get<B>(a)=stod(argv[B+1]);
		validate(a);
		map<double,double> pl{}; double value{};
		while(cin>>value) pl[value]++; // reads PLs counting multiplicities
		V b{}; V w{}; b.reserve(pl.size()); w.reserve(pl.size());
		for(const auto& plMultiplicity:pl) {
			b.push_back(plMultiplicity.first);  // boundaries are unique PL
			w.push_back(plMultiplicity.second); // weights are multiplicities
		}
		HDistr d{};
		if(get<DISTR>(a)=='l')d={L{},b.cbegin(),b.cend(),w.cbegin()};
		else if(get<DISTR>(a)=='c')d={C{},b.cbegin(),b.cend(),w.cbegin()};
		else d={D{},b,w};
		mt19937 e(get<SEED>(a));
		SampleDistribution aDistr{}, gDistr{};
		for(size_t j{};j<get<PS>(a);j++){
			const auto an{iteral(get<TS>(a),get<ACC>(a),
			[&](const auto& acc)
			{
				const auto Apl{d(e)}; // pseudo random PL
				const auto n{acc<0.0?0.0:floor(acc*get<B>(a)/get<MARGIN>(a))};
				if(get<B>(a)<=1.0) return acc+Apl*n;
				if(acc<=0.0) return acc;
				return Apl*n+acc*(1.0-(get<B>(a)-1.0)*get<RATE>(a)/(365.0*
				get<DAILY>(a)));
			})};
			const auto gn{an<=0.0?0.0:log(an/get<ACC>(a))/(numbers::ln2*
			get<TS>(a))};
			aDistr[an]++; gDistr[gn]++;
		}
		if(get<OUT>(a)=='a'){cout<<toEqualIS(aDistr,get<IS>(a));return 0;}
		if(get<OUT>(a)=='s'){cout<<toEqualIS(gDistr,get<IS>(a));return 0;}
		const auto aAvg{average(aDistr)}, gAvg{average(gDistr)},
		aDev{stdev(aDistr)}, gDev{stdev(gDistr)};
		constexpr int WI{7},WI2{3},WD{20},WD2{10},P{12};
		cout<<get<DISTR>(a)<<" "
		<<setw(WI)<<get<PS>(a)<<" "
		<<setw(WI)<<get<SEED>(a)<<" "
		<<setw(WD2)<<setprecision(P)<<get<ACC>(a)<<" "
		<<setw(WD2)<<setprecision(P)<<get<MARGIN>(a)<<" "
		<<setw(WI2)<<get<TS>(a)<<" "
		<<setw(WD2)<<setprecision(P)<<get<RATE>(a)<<" "
		<<setw(WI2)<<get<DAILY>(a)<<" "
		<<setw(WI2)<<get<IS>(a)<<" "
		<<setw(WD2)<<setprecision(P)<<get<B>(a)<<" "
		<<setw(WD)<<setprecision(P)<<(*aDistr.cbegin()).first<<" "
		<<setw(WI)<<(*aDistr.cbegin()).second<<" "
		<<setw(WD)<<setprecision(P)<<aAvg<<" "
		<<setw(WD)<<setprecision(P)<<aDev<<" "
		<<setw(WD)<<setprecision(P)<<(*aDistr.crbegin()).first<<" "
		<<setw(WI)<<(*aDistr.crbegin()).second<<" "
		<<setw(WD)<<setprecision(P)<<(*gDistr.cbegin()).first<<" "
		<<setw(WI)<<(*gDistr.cbegin()).second<<" "
		<<setw(WD)<<setprecision(P)<<gAvg<<" "
		<<setw(WD)<<setprecision(P)<<gDev<<" "
		<<setw(WD)<<setprecision(P)<<(*gDistr.crbegin()).first<<" "
		<<setw(WI)<<(*gDistr.crbegin()).second<<" "
		<<endl;
	}
	catch(const exception& e){cerr<<e.what()<<endl;return 1;}
	catch(...){cerr<<"Uknown exception"<<endl;return 2;}
	return 0;
}
