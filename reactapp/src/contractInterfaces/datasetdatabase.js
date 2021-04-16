import web3 from './web3';
import { datasetDatabase} from "../abi/abi";

const address = '0x7360e8eC73143658cfD9b53adDC95F9Bd99bc2d2';

export default new web3.eth.Contract(datasetDatabase, address);