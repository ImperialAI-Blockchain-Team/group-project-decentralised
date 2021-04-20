import web3 from './web3';
import { datasetDatabase} from "../abi/abi";

const address = '0x52f6839425576cBF9AD91E6A92ADe361f388FF9D';

export default new web3.eth.Contract(datasetDatabase, address);
