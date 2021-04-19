import web3 from './web3';
import { datasetDatabase} from "../abi/abi";

const address = '0xC74C6A4Df6C9Fb3449D31D824c33500e737b130F';

export default new web3.eth.Contract(datasetDatabase, address);