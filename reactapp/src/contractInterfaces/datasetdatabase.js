import web3 from './web3';
import { datasetDatabase} from "../abi/abi";

const address = '0x8f016f8ee1144Bf8F398D15f762826155C3D1610';

export default new web3.eth.Contract(datasetDatabase, address);
