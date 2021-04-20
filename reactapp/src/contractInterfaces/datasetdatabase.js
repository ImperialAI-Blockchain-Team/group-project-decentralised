import web3 from './web3';
import { datasetDatabase} from "../abi/abi";

const address = '0xF57c008F01F8Fbb3Bc5bD59308B73D874C5F2553';

export default new web3.eth.Contract(datasetDatabase, address);
