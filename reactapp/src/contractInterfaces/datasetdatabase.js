import web3 from './web3';
import { datasetDatabase} from "../abi/abi";

const address = '0x15AcC3661FD466dC10229F7f97F4D7F536679d6F';

export default new web3.eth.Contract(datasetDatabase, address);
