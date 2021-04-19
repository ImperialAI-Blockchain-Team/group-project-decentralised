import web3 from './web3';
import { datasetDatabase} from "../abi/abi";

const address = '0x9a6D6b9c1D067Ba109b3684B4597e864C3bcf797';

export default new web3.eth.Contract(datasetDatabase, address);