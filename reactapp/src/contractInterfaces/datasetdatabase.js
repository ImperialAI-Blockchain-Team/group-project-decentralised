import web3 from './web3';
import { datasetDatabase} from "../abi/abi";

const address = '0x61C9049D9c40fe64BF382a32eF419501F245912C';

export default new web3.eth.Contract(datasetDatabase, address);
