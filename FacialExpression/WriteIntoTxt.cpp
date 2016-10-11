/************************************************************************/
/*               write the features into txt file                       */
/************************************************************************/



#include "WriteToTxt.h"

int WriteIntoTxt(int filetowrite[2400]){
	cout<<"writing to txt..."<<endl;
	std::ofstream ofile;
	ofile.open("2.txt",ios::out|ios::app);
	for(int i=0;i<2400;i++){
		if(i==0) ofile<<"-1 "<<i<<":"<<filetowrite[i]<<" ";
		if(i!=2399&&i!=0) ofile<<i<<":"<<filetowrite[i]<<" ";
		if(i==2399) {ofile<<i<<":"<<filetowrite[i]<<'\n';}
	}
	ofile.close();
	cout<<"write to txt success..."<<endl;
	return 1;
}