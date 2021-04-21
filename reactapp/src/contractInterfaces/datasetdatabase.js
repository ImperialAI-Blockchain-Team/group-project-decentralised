import web3 from './web3';
import { datasetDatabase} from "../abi/abi";

const address = '0x853b3d555Ed8b9a369d35175F072a38b7492441c';

export default new web3.eth.Contract(datasetDatabase, address);
